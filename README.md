# dlsr

The harrypotter sorting hat as a logistic regression

We have a bunch of data for students and their scors in the various subjects at hogwards: "Flying", "Arithmancy", "Muggle Studies" ...

This data needs to analysed to see what "features" or subjects to keep.

Then we build a classifier using a logistic regression, and assign students to one of the 4 houses based of their grades.

As for "ft_linear_regression" the [google ml crashcourse](https://developers.google.com/machine-learning/crash-course/logistic-regression) is great.

# Getting started

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for python package managment and venv. It's a simple install script from their website, and it's fast!

Then you can run:
```bash
./make.sh fetch
# .. this will fetch the project data, (so long as the urls has not changed)

./make.sh venv
# .. this will make the venv, (required uv to be installed)

source .venv/bin/activate
# .. this will activate the python venv for you shell instance

./make.sh jupyter
# .. this will launch the jupyter server so you can view the code notebooks (`.ipynb`) files. 
```

# Encoding dates

It's complicated and you can extract various bit of data from them, it could be: age, or the month as a cyclic thing, you could imaging it's related to the astological sign for instance. This article talks about [how to reprisent cyclic time based information](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/).


# TODO

- [ ] error handelling everywhere
- [x] bonus fields for describe
- [x] readme with setup instructions, (install uv, ./make.sh etc..)
- [ ] for train
- [x]  - implement the train algo
- [x]  - decide what to do when training with empty data in one column
- [x]  - how to handle data normalization
- [x]  - do we use numpy? pandas? or just raw python
- [x]  - sigmoid with very big or small number, how should it be handled
- [ ] arguemnt for train and predict
- [ ] bonus algo, stochastic gradient descent
- [ ] bonus optimization algo, batch, mini batch
- [ ] bonus find 3rd bonus optimization algo
- [ ] final tests from shool repo and fresh folder
- [ ] push and eval