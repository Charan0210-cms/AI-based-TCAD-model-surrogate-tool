# AI-based-TCAD-model-surrogate-tool
Machine learning model for Nanosheet FETs that predicts device performance (ION, IOFF, Cgg) in milliseconds, enabling faster design and optimization replacing traditional TCAD tools.

I worked on building a faster alternative to TCAD simulations for nanosheet FET devices using machine learning. Instead of running full physics-based simulations every time (which can take a long time), the idea was to train a model that can predict device behavior almost instantly.

How I approached it

### 1. Data generation

Since I didn’t have access to real TCAD data, I created a synthetic dataset that follows basic device physics trends.
I varied parameters like:

* corner radius
* gate length
* oxide thickness
* doping
* bias conditions (Vgs, Vds)

and generated corresponding outputs like current and capacitance.

### 2. Model development

I trained an XGBoost regression model to learn the relationship between these inputs and outputs.

The model predicts:

* ION (drive current)
* IOFF (leakage current)
* Cgg (gate capacitance)

The goal was to capture the nonlinear behavior between geometry and electrical performance.

### 3. Speed advantage

Once trained, the model can predict results in milliseconds instead of minutes like TCAD.
This makes it much easier to explore different device configurations quickly.

### 4. Optimization

I added a simple optimization loop to find the best device parameters.
For example, I looked at maximizing the ratio of ION to Cgg, which reflects a common tradeoff in device design.

### 5. Interface

I built a small Streamlit app where you can:

* change device parameters
* instantly see how performance changes

This makes it easier to interact with the model instead of just running scripts.

## Why this is useful

The main idea is to speed up the design process.
Instead of running multiple expensive simulations, you can use a trained model to quickly:

* estimate performance
* compare designs
* narrow down good configurations



