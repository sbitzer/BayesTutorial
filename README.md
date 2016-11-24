# BayesTutorial
A collection of notebooks and scripts demonstrating various conceptual aspects of Bayesian inference.

## CohensCongressExample notebook
Picks up on an example which points to a possible misinterpretation when transforming natural
language into statements about probability, especially applied to statistical hypothesis testing.
Goes on to discuss the probability that a research finding is true given a significant 
hypothesis test (positive predictive value, PPV). Exemplifies how this probability depends on 
the base rate of research hypotheses being true and the power of the test.

## GaussInference notebook
Some examples of Bayesian inference with simple Gaussian models:
- estimating a mean from a single, one-dimensional data point 
  (demonstrates how estimates depend on prior and likelihood)
- inferring two hidden causes from a single, one-dimensional data point 
  (demonstrates two-dimensional Gaussians, relationship between prior and likelihood, 
  how estimated causes depend on assumed prior correlations and model)

## estimating_a_mean script
Demo showing Bayesian estimation of a two-dimensional mean as data points arrive sequentially.
Can demonstrate:
- narrowing of posterior for more data points
- that posterior will include true mean most of the time (calibrated)
- that posterior will not include true mean most of the time (uncalibrated)
- that posterior is influenced by prior, but data overwrites that influence

