- Train models based on different correspondence and architecture settings:
	1. both correspondence and architecture is fixed (initialization, data, and noise from SGD introduces variance)
	2. correspondence fixed, architecture varies
	3. correspondence varies (within the same layer, we interchange one-to-many and many-to-one mappings; we do not permute the features across different layers), architecture is fixed
	4. both architectures and correspondences are varied.
- Synthesize a common dataset to get the representations and perform interventions.
	- Check model performance on this dataset, to make sure that there is no distribution shift.
	- Get the representations of the model (not) under intervention
- Perform alignment between the representations of any set of models.