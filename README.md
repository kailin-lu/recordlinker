# recordlinker 

Work in progress for research asssistantship with Professor Suresh Naidu at Columbia University 

### Example 

```buildoutcfg
from recordlinker.models import VAE

# Train a variational autoencoder model
vae = VAE()
model, encoder, decoder = vae.train(namesA, namesB) 

# Reduce number of candidate links through blocking
blocker = Blocker(namesA, namesB)

blocks = blocker.block(autoencoder_col='last1',
                       autoencoder_colB='last2', 
                       autoencoder_model_path=model_path)

# Create comparison features 
comparer = Comparer(blocker)
comparer.compare_autoencoder(colA='lname1915', colB='lname1940', model_path=compare_model_path)
comparer.compare_jarowinkler(colA='lname1915', colB='lname1940')

features = comparer.discretize({'autoencoder': 0.88, 
                                'jarowinkler': 0.88,
                                'jarowinkler-first': 0.88,
                                'product-autoencoder-jarowinkler': .8}, 
                              binary=True)

```

### Tools

`preprocess`

`models`

`blocking`

### Notebooks 

* TrainingExample - train encoder models 
* AccuracyComparison - compare accuracy of 
  Jaro Winkler distance vs. autoencoder distance
* Blocking - compute blocking metrics
* Linking - build comparison features, linking using recordlinkage 

### Requirements 

* numpy 
* pandas 
* scikit-learn 
* keras 

### Included datasets 

Iowa Census - James F. 

Union Army 

### References 







