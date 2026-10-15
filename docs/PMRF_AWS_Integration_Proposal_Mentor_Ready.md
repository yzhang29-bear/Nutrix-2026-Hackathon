# AWS Integration Proposal for Predict My Glucose (PMRF)

## Goal

Use AWS to turn our existing **Personalized Metabolic Response Function
(PMRF)** into a secure, explainable, production-ready AI application
**without replacing the core prediction model**.

Our core challenge remains:

> **Can seven days of multimodal wearable and meal data teach an
> algorithm enough about an individual to predict how that person will
> respond to a new meal?**

The proposed AWS integration adds a second product-level question:

> **How can we turn that personalized prediction model into a scalable,
> explainable, and responsible AI application?**

------------------------------------------------------------------------

## 1. What We Already Have

The current PMRF pipeline already performs the main predictive task.

### Inputs

-   Meal photo
-   Recent CGM history
-   Heart rate
-   Medication information, when available
-   Days 1--7 personal history

### Prediction pipeline

-   **MobileCLIP-S2** for meal-image representation
-   Nutrition/photo feature estimation
-   **Personalized LightGBM**
-   **In-context transformer**
-   Ensemble prediction
-   TreeSHAP / similar-meal explanations
-   Uncertainty estimation

### Outputs

-   Glucose at 30, 60, 90, and 120 minutes
-   Full predicted glucose trajectory
-   Peak glucose
-   Peak rise
-   Time to peak
-   0--120 min iAUC
-   80% prediction interval
-   Probability of high glucose peaks
-   Important contributing factors
-   Similar past meals
-   Personalized what-if scenarios

**Key principle:** AWS should support deployment, orchestration,
explanation, safety, and scalability. We should **not replace the
existing PMRF prediction model with an LLM**.

------------------------------------------------------------------------

## 2. Current Implementation Status from PR #2

PR #2 (`feature/model` → `main`) materially changes the AWS discussion: the PMRF core is no longer only a research model. The branch now includes a **CPU-only FastAPI prediction service, Docker image, versioned model store, AWS deployment guide, API contract, S3 integration, and acceptance tests**.

### Current model / service facts

- **Model version:** `v4-20261010`
- **Runtime:** CPU-only FastAPI container
- **Target task size:** 2 vCPU / 4 GB
- **Memory:** about 1.0 GB in load testing
- **First-week personalization state build:** about 2–3 seconds
- **No per-user LightGBM retraining:** `/history` builds a first-week state containing history meals, photo embeddings, personal profile features, and the user's labeled-photo retrieval library; inference then uses that state.
- **Model store:** 115 files, 466 MB, stored in encrypted/private/versioned S3 under `s3://nutrix-models/models/v4-20261010/`

### Validation and load-test evidence

- **44 people / 402 day-8+ meals:** unseen-meal peak-rise error = **19.8 mg/dL**
- **Golden test:** 176 / 176 predictions matched the technical-report outputs
- **End-to-end API flow:** 13 / 13 checks passed
- **S3 photo + model-download tests:** passed
- **Single client:** median latency **252 ms**, p99 **374 ms**, about **3.8 predictions/s**
- **Four clients:** same overall throughput, median latency about **1.2 s**
- **Container startup:** about **3 s**

This means the mentor meeting should focus less on "how do we host the model at all?" and more on whether the current AWS pattern is the right level of complexity for the hackathon and how to add the strongest explainability layer next.

------------------------------------------------------------------------

## 3. Current AWS Deployment Architecture

The deployment guide in PR #2 specifies the following path:

``` text
App
  |
  | HTTPS
  v
API Gateway (HTTP API)
  |
  | Cognito JWT authorizer
  v
VPC Link -> ALB
  |
  v
ECS Fargate
  |
  v
PMRF FastAPI Service
```

Supporting AWS services:

``` text
Meal photos
   |
   +--> Presigned S3 uploads

Per-person signals + first-week state
   |
   +--> Amazon EFS mounted at /data

Model versions
   |
   +--> Amazon S3 -> rolling ECS deployment

Operational telemetry
   |
   +--> CloudWatch request logs
```

### Concrete deployment configuration

- **ECS Fargate:** Linux x86_64, 2 vCPU, 4 GB, port 8080, one uvicorn worker per task
- **API Gateway:** `/v1/*` and `/healthz`; Cognito JWT authorizer; first-week upload is roughly 1–2 MB
- **S3:** presigned meal-photo uploads; encryption enabled; public access blocked; model versions can also be loaded from S3
- **EFS:** per-person raw signals and first-week state, roughly 1–3 MB per person, encrypted at rest
- **CloudWatch:** one JSON request log per call containing method, route, status, and latency; the service does **not** log glucose values or meal photos
- **Scaling:** target CPU 60%; one task serves roughly 3.5–3.8 predictions/s

### App flow already defined by the API

``` text
1. Upload meal photo -> presigned S3 PUT
2. POST /history -> build first-week personalization state (~2–3 s)
3. POST /signals -> append newer CGM / HR / meals / meds
4. POST /predict -> return prediction + explanations + what-if scenarios
5. GET /profile -> return personalization, diet guide, and metabolic profile
```

### Privacy / security already represented in the deployment design

- Authenticated API path through Cognito / API Gateway
- Encryption at rest for S3 and EFS
- TLS at API Gateway
- Pseudonymous person IDs recommended instead of names/emails
- Non-root container and input validation
- CloudWatch logs exclude glucose values and photos
- If real users' health data is later processed, the deployment guide recommends using HIPAA-eligible services and an AWS BAA

### Important implementation limits

- Predictions require a CGM reading within 30 minutes before the meal
- The diet guide is fully personal only when standardized first-week test meals are available; otherwise the service falls back to group-average intervention effects
- Person data is currently file-based on EFS; high write concurrency per person may eventually justify DynamoDB or another managed store
- The current model was developed on CGMacros; it should be retrained on the official challenge dataset when available

------------------------------------------------------------------------

## 4. Amazon Bedrock --- Explanation Layer

Bedrock should **not predict glucose directly**.

Instead, the PMRF model first produces structured numerical predictions
and explanation features.

Example:

``` text
Peak rise: +42 mg/dL

Contributing factors:
High carbohydrate content        +18 mg/dL
Personal carbohydrate sensitivity +11 mg/dL
Elevated baseline glucose          +7 mg/dL
Recent activity                    -4 mg/dL
```

These structured results can be passed to a Bedrock-hosted model to
generate a user-friendly explanation:

> Your glucose is predicted to rise approximately 42 mg/dL after this
> meal. The meal's carbohydrate content is the largest contributor.
> Based on your first week of data, you also appear to have a
> stronger-than-average response to carbohydrate-rich meals.

This creates a useful separation:

``` text
PMRF
Predictive AI
"What will happen?"
       |
       v
Amazon Bedrock
Generative AI
"How do we explain it?"
```

This keeps the medically relevant numerical prediction grounded in our
trained model while using generative AI for communication.

------------------------------------------------------------------------

## 5. Amazon Bedrock Guardrails --- Responsible AI

This is especially relevant because our application is health-related.

Bedrock Guardrails should be configured **at the same time as the Bedrock explanation layer**, rather than added at the end. They can help prevent the generative explanation layer from producing unsupported medical claims or inappropriate recommendations.

For example, the system should avoid outputs such as:

> "You definitely have diabetes."

or

> "Stop taking your medication."

Instead, the application should communicate uncertainty and clearly
distinguish predictive insights from medical advice.

Proposed flow:

``` text
PMRF Prediction
      |
      v
Bedrock Explanation
      |
      v
Bedrock Guardrails
      |
      v
Safe User-Facing Response
```

This strengthens the project's **Responsible AI** story.

------------------------------------------------------------------------

## 6. Medical Knowledge–Grounded Interpretation & Meal-Level GI/GL Estimation

Explainability should be a central feature of the application rather than an add-on. In addition to explaining which model features contributed to a prediction, we can connect the predicted glucose response to curated medical and nutritional evidence while explicitly communicating uncertainty.

### A. Medical Knowledge–Grounded Interpretation

Build a curated medical knowledge base using peer-reviewed literature, validated clinical references, CGM research, project methodology, and model limitations. After PMRF predicts an individual's postprandial glucose response, the system can extract clinically meaningful CGM response features and retrieve relevant evidence to provide **possible physiological interpretations**.

``` text
PMRF Prediction
      |
      v
Predicted Glucose Response
Trajectory / Peak / iAUC / Variability
      |
      v
Extract CGM Response Features
      |
      v
Amazon Bedrock Knowledge Base / RAG
      |
      v
Evidence-Grounded Interpretation
      |
      v
Evidence Support / Grounding Score
      |
      v
User Report + Supporting Citations
```

The report could discuss patterns such as unusually high postprandial excursions, prolonged glucose elevation, delayed recovery, or high glycemic variability. These outputs should be framed as **model-based interpretations rather than clinical diagnoses**. The explanation should not report an LLM-generated self-confidence score. Instead, we should expose an **evidence-support / grounding score** based on how well the generated explanation is supported by the PMRF outputs and retrieved references, alongside the PMRF's own predictive uncertainty.

CGM-derived phenotyping approaches, such as published work on *glucotypes* and patterns of glucose dysregulation, could also be explored to characterize an individual's glucose-response phenotype. This should be presented as exploratory metabolic characterization, not as a tool for diagnosing or assigning a diabetes stage.

### B. Meal-Level Glycemic Index (GI) and Glycemic Load (GL) Estimation

Extend the existing meal-image workflow so that it estimates not only food identity and macronutrients, but also the overall glycemic characteristics of the meal.

``` text
Meal Image
     |
     v
Food Recognition
     |
     v
Food Segmentation + Portion Estimation
     |
     v
Estimated Available Carbohydrate per Food
     |
     v
Validated GI Database Lookup
     |
     v
Food-Level GI + GL
     |
     v
Meal-Level Weighted GI + Total GL
     |
     v
Explanation Layer / User Report
```

The system could reference validated resources such as international GI tables to estimate the GI of carbohydrate-containing foods and calculate approximate food-level and meal-level glycemic load from estimated portions and available carbohydrate. These values should be reported as estimates because image-based food identification, portion estimation, preparation method, mixed-meal effects, and GI-table matching all introduce uncertainty. **For the hackathon MVP, GI/GL should remain explanation-only and should not feed back into PMRF as a predictive feature.** If we later want GI/GL to become a model input, the model should be retrained and revalidated with that feature explicitly included.

### C. Combined Explainability Workflow

The two additions can be integrated with the existing SHAP and similar-meal explanations:

``` text
                    Meal Image
                        |
                        v
          Food + Portion Recognition
                        |
                        v
        Nutrition Estimation
                        |
                        v
                    PMRF Model
                        |
                        v
        Personalized CGM Prediction
                        |
       +----------------+----------------+
       |                                 |
       v                                 v
Model Explainability              Medical Knowledge Base
SHAP / Similar Meals              Literature / CGM Evidence
       |                                 |
       +----------------+----------------+
                        |
                        v
           Evidence-Grounded Explanation
                        |
                        v
      Evidence Support / Grounding Score
                        |
                        v
        User Report + Supporting Citations
                        ^
                        |
       GI/GL Estimate (explanation-only)
```

The resulting user experience should answer three questions:

1. **What is predicted to happen?** — personalized glucose trajectory, peak, iAUC, and uncertainty.
2. **Why is it predicted?** — meal composition, estimated GI/GL, SHAP factors, similar meals, and personal metabolic context.
3. **What may explain the pattern physiologically, and how well is that explanation supported?** — evidence-grounded interpretation with citations plus an evidence-support / grounding signal. Predictive uncertainty remains a separate PMRF output.

This creates a stronger explainability story by connecting **food → glycemic characteristics → personal context → predicted response → evidence-grounded interpretation**.

------------------------------------------------------------------------

## 7. Optional: Additional Bedrock Knowledge Base / RAG Interactions

If time permits, we can add a small curated knowledge base containing:

-   Project methodology
-   CGM interpretation documentation
-   Relevant nutrition literature
-   Model limitations
-   Approved health-information resources

Then a question such as:

> "Why might adding protein change my glucose response?"

could follow:

``` text
User Question
      |
      v
Bedrock Knowledge Base
      |
      v
Retrieve Relevant Evidence
      |
      v
LLM Explanation
      |
      v
Answer + Citation
```

Additional conversational RAG features are **Phase 2**. The core medical knowledge-grounded interpretation can be implemented first with a small, curated evidence set.

------------------------------------------------------------------------

## 8. Kiro for Development

Kiro would be a **development tool**, not part of the deployed
prediction architecture.

We can use Kiro's spec-driven development workflow to turn our existing
requirements and API contract into implementation tasks:

``` text
PMRF Technical Report
        |
        v
API Specification
        |
        v
Kiro
        |
        v
Requirements
        |
        v
Design
        |
        v
Tasks
        |
        v
Implementation + Tests
```

Good Kiro tasks for our project could include:

-   FastAPI backend
-   AWS S3 integration
-   API Gateway integration
-   Docker configuration
-   Bedrock explanation service
-   Guardrail integration
-   Unit/integration tests
-   Frontend/backend API integration

------------------------------------------------------------------------

## 9. Recommended Hackathon MVP from the Current PR #2 Baseline

We should avoid adding AWS services only for the sake of having more AWS
logos.

The first version can stay relatively simple:

``` text
                    Web App
                       |
                       v
                 API Gateway
                       |
                       v
                 Backend API
                  /         \
                 v           v
            Amazon S3      PMRF Model
           Meal / Data     Prediction
                              |
                              v
                       Amazon Bedrock
                        Explanation
                              |
                              v
                      Bedrock Guardrails
                              |
                              v
                       Prediction UI
```

### Implementation Priority

**Priority 1 --- Finish the current service path** 1. Confirm the PR #2 FastAPI service works in the HealthHack AWS environment 2. Connect the frontend to presigned S3 uploads and `/predict` 3. Show the prediction curve, 80% band, explanation factors, and profile data

**Priority 2 --- Explainability & Responsible AI** 4. Add Bedrock-generated explanations **together with Bedrock Guardrails** 5. Add a small curated medical knowledge base for evidence-grounded interpretation 6. Add citations and an evidence-support / grounding signal

**Priority 3 --- If Time Permits** 7. Add food-level and meal-level GI/GL estimation as an **explanation-only** feature 8. Expand conversational RAG 9. SageMaker/MLOps 10. More advanced agent workflows

------------------------------------------------------------------------

## 10. Product Story

The architecture gives us three complementary AI layers:

### Predictive AI

**PMRF**

> What will this meal do to **my** glucose?

### Generative AI

**Amazon Bedrock**

> Why is this response predicted, and how can the result be explained
> clearly?

### Responsible AI

**Bedrock Guardrails + PMRF uncertainty**

> How can we communicate the result safely and transparently?

This is stronger than simply saying that we "used AWS."

------------------------------------------------------------------------

## 11. Proposed Demo Flow

A clean demo could look like this:

### Step 1 --- Personal history

The system has learned from the user's first seven days of:

-   CGM
-   meals
-   meal photos
-   heart rate
-   medication/context

### Step 2 --- New meal

The user uploads a Day 8 meal photo.

### Step 3 --- PMRF prediction

The model predicts:

``` text
30 min       146 mg/dL
60 min       161 mg/dL
90 min       153 mg/dL
120 min      139 mg/dL

Peak          164 mg/dL
Peak rise     +47 mg/dL
Time to peak   52 min
80% interval   Displayed in UI
```

The UI also displays the predicted trajectory and uncertainty band.

### Step 4 --- Personalized and evidence-grounded explanation

Bedrock combines PMRF/SHAP output, similar meals, and retrieved medical evidence to produce a concise explanation of the major factors and plausible physiological interpretation. The UI should display supporting citations and an **evidence-support / grounding signal**. This is separate from the PMRF 80% prediction interval, which communicates predictive uncertainty.

### Step 5 --- Meal GI/GL and What-if

If time permits, the UI can show estimated meal-level GI/GL as **explanation-only context** and allow the user to compare scenarios such as:

-   As photographed
-   Half the carbohydrates
-   Add protein
-   Add fiber
-   Double portion

### Step 6 --- Safety

Guardrails keep the explanation within the scope of decision support and
prevent unsupported diagnostic or medication advice.

------------------------------------------------------------------------

## 12. Why AWS Adds Value

AWS is not replacing our machine-learning contribution.

It converts the existing research prototype into a more complete AI
product:

``` text
Research Model
      |
      v
Personalized Prediction
      |
      v
AWS Deployment
      |
      v
Generative Explanation
      |
      v
Responsible AI
      |
      v
Production-Ready Application
```

### Core message

> **Our innovation is the personalized multimodal glucose-response
> model. AWS provides the infrastructure and generative-AI layer needed
> to turn that model into a scalable, explainable, and responsible
> application.**

------------------------------------------------------------------------

## Proposed Next Step

For the next implementation sprint, the PMRF service and AWS deployment path already exist in PR #2. Focus on:

**Frontend integration with the PR #2 API + Bedrock Explanation + Bedrock Guardrails**

Once the PR #2 predictor is running end-to-end with the frontend, add a **small curated medical Knowledge Base / RAG layer** for grounded interpretation and citations. Broader conversational RAG remains optional. GI/GL should remain explanation-only for the hackathon and can be cut before the medical RAG if scope becomes tight. SageMaker/MLOps and more advanced agent functionality are later extensions.
