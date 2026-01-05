# dlsr

The harrypotter sorting hat as a logistic regression

We have a bunch of data for students and their scors in the various subjects at hogwards: "Flying", "Arithmancy", "Muggle Studies" ...

This data needs to analysed to see what "features" or subjects to keep.

Then we build a classifier using a logistic regression, and assign students to one of the 4 houses based of their grades.

As for "ft_linear_regression" the [google ml crashcourse](https://developers.google.com/machine-learning/crash-course/logistic-regression) is great.

## Encoding dates

It's complicated and you can extract various bit of data from them, it could be: age, or the month as a cyclic thing, you could imaging it's related to the astological sign for instance. This article talks about [how to reprisent cyclic time based information](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/).


## TODO

- [ ] error handelling in describe
- [ ] bonus fields for describe
- [ ] readme with setup instructions, (install uv, ./make.sh etc..)
- [ ] for train
- [ ]  - decide what to do when training with empty data in one column
- [ ]  - how to handle data normalization
- [ ]  - do we use numpy? pandas? or just raw python
- [ ]  - sigmoid with very big or small number, how should it be handled