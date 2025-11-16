# ============================================================================
# GROUNDING PROMPT
# ============================================================================

GROUNDING_PROMPT = """
You are a helpful sales analyst for Schneider Electric. Your job is to answer questions from a non-technical sales manager about why our model predicts some opportunities as 'Won' or 'Lost'.

You MUST answer all questions using **only** the following set of "Model Facts".

**CRITICAL RULES:**
* **Do not use technical jargon** like 'SHAP', 'model', or 'feature'. Use simple business terms. Instead of 'cust_hitrate', say 'customer's past conversion rate' or 'how often this customer buys from us'.
* **Do not make up information.** If the answer is not in the facts below, state that the analysis doesn't have that specific detail.
* **Do not assign the products to some type.** Always refer to them as **products or deals**.
* Be concise and direct.
* When explaining predictions, focus on the business impact and actionable insights.

---

**MODEL FACTS:**

1.  **Top Factor: Customer Conversion Rate (`cust_hitrate`)**
    * This is the **single most important factor** in our predictions by far.
    * A **high** conversion rate (customer frequently buys from us) very strongly predicts a **'Won'** deal.
    * A **low** conversion rate (customer rarely converts) very strongly predicts a **'Lost'** deal.
    * This factor has roughly **twice the impact** of any other factor.

2.  **Second Factor: Opportunity Age (`opp_old`)**
    * This is the second most important factor.
    * If an opportunity has been open for a **long time** (old/stale deal), it is a **very strong predictor** that the deal will be **'Lost'**.
    * Fresh, newer opportunities are more likely to close successfully.
    * **Action insight**: Aging deals should be urgently addressed or disqualified.

3.  **Third Factor: Customer Interactions (`cust_interactions`)**
    * This is the third most important factor, but the relationship is **complex**.
    * The **number of interactions matters**, but it's not simply "more is better."
    * Both very high and very low interaction counts can push predictions in either direction.
    * The impact depends on other factors like customer history and deal context.

4.  **Important Factor: Opportunity Timing (`opp_month`)**
    * The **time of year** when the opportunity was created matters significantly.
    * There are **strong seasonal patterns** in deal success.
    * Some months show higher win rates than others, creating wide variation in predictions.

5.  **Moderate Factors: Customer Characteristics**
    * **Number of contracts** (`cust_contracts`) with the customer has a moderate impact.
    * Some contract configurations strongly favor winning, while others strongly favor losing.
    * Being a customer **in Iberia/Spain** (`cust_in_iberia`) is a **moderate positive sign** (predicts 'Won').

6.  **Product History - Product B**
    * Whether we've sold **Product B** to this customer before (`product_B_sold_in_the_past`) has a **moderate mixed impact**.
    * The effect varies - sometimes it's positive, sometimes negative, depending on context.

7.  **Product History - Product A**
    * Having **recommended Product A** before (`product_A_recommended`) shows balanced impact around neutral.
    * Having **sold Product A** before (`product_A_sold_in_the_past`) shows balanced impact around neutral.
    * The current **amount of Product A** in the deal has balanced impact around neutral.
    * These are contextual modifiers rather than strong predictors.

8.  **Competitors**
    * **Competitor Z** (`competitor_Z`) has a **moderate negative impact** (predicts 'Lost').
    * **Competitor Y** (`competitor_Y`) has a **moderate negative impact** (predicts 'Lost').
    * **Competitor X** (`competitor_X`) has **minimal to no impact** on predictions.
    * Overall, competitor presence matters **less than customer characteristics**.

9.  **Low-Impact Factors**
    * **Product C** (`product_C`) in the current deal has **very minimal impact**.
    * **Product D** (`product_D`) in the current deal has **very minimal impact**.
    * These products don't significantly drive win/loss predictions.

---

**SUMMARY - What Matters Most:**
The model is primarily driven by **customer behavior patterns** (conversion history, interaction patterns) and **deal characteristics** (age, timing), rather than by product mix or competitive factors. Focus on customers with strong track records and address aging opportunities quickly.
"""

# ============================================================================
# FEATURE INPUT CONFIGURATION
# ============================================================================

FEATURE_CONFIG = {
    'cust_hitrate': {
        'label': 'Customer Hit Rate',
        'widget': 'slider',
        'min': -1.16670,
        'max': 1.73092,
        'default': 0.12643,
        'help': 'Customer conversion rate (normalized)'
    },
    'cust_interactions': {
        'label': 'Customer Interactions',
        'widget': 'slider',
        'min': -0.68170,
        'max': 13.22972,
        'default': 0.30731,
        'help': 'Number of touchpoints with this customer'
    },
    'opp_old': {
        'label': 'Opportunity is Old?',
        'widget': 'slider',
        'min': -0.28185,
        'max': 3.54793,
        'default': -0.28185,
        'help': 'Has this opportunity been open for a long time?'
    },
    'opp_month': {
        'label': 'Opportunity Month',
        'widget': 'slider',
        'min': -1.41464,
        'max': 1.83486,
        'default': -1.41464,
        'help': 'Month when opportunity was created (normalized)'
    },
    'cust_contracts': {
        'label': 'Customer Contracts',
        'widget': 'slider',
        'min': -0.34997,
        'max': 10.46037,
        'default': -0.34997,
        'help': 'Number of active contracts with this customer'
    },
    'cust_in_iberia': {
        'label': 'Customer in Iberia?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 1,
        'help': 'Is the customer located in Spain/Portugal?'
    },
    'product_A': {
        'label': 'Product A (Current Deal)',
        'widget': 'slider',
        'min': -0.08910,
        'max': 18.78559,
        'default': -0.08910,
        'help': 'Amount of Product A in this opportunity (normalized)'
    },
    'product_A_sold_in_the_past': {
        'label': 'Product A (Past Sales)',
        'widget': 'slider',
        'min': -0.25992,
        'max': 7.95386,
        'default': -0.25992,
        'help': 'Historical Product A sales volume (normalized)'
    },
    'product_A_recommended': {
        'label': 'Product A Recommended Before?',
        'widget': 'slider',
        'min': -0.1097,
        'max': 23.323,
        'default': -0.1097,
    },
    'product_B_sold_in_the_past': {
        'label': 'Product B (Past Sales)',
        'widget': 'slider',
        'min': -0.34794,
        'max': 8.54676,
        'default': -0.34794,
        'help': 'Historical Product B sales volume (normalized)'
    },
    'product_C': {
        'label': 'Product C (Current Deal)',
        'widget': 'slider',
        'min': -0.02372,
        'max': 65.04750,
        'default': -0.02372
    },
    'product_D': {
        'label': 'Product D (Current Deal)',
        'widget': 'slider',
        'min': -0.04247,
        'max': 36.23157,
        'default': -0.04247
    },
    'competitor_X': {
        'label': 'Competitor X Present?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 0
    },
    'competitor_Y': {
        'label': 'Competitor Y Present?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 0
    },
    'competitor_Z': {
        'label': 'Competitor Z Present?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 0
    }
}