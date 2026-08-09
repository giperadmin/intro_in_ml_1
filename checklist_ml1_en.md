# Project Review - ML1 Introduction

Source PDF: `Чек лист ML1 на англ.pdf`

## Project Info

| Field | Value |
| --- | --- |
| Type of project | Individual |
| Duration | 30 min |
| Passed Peer Reviews | 2/3 |

## Git Project

Repository: `ssh://git@git-ssh.21-school.ru:2222/students_repo/jackip...`

Student: `jackipax`

Level: `4`

## About

### Introduction

School 21 methodology only makes sense if peer-to-peer reviews are taken seriously. Please read all the guidelines carefully before starting the review.

Please remain courteous, polite, respectful, and constructive in all communications during this review.

Point out possible flaws in the person's work and take the time to discuss and debate them.

Keep in mind that sometimes there may be differences in interpretation of tasks and scope of functionality. Please remain open to each other's vision.

### Guidelines

Evaluate only the files located in the `src` folder of the student's or group's GIT repository.

If you have not finished the project yet, it is compulsory to read the entire instruction before starting the review.

Ensure to start reviewing a group project only when the team is present in full.

Use special flags in the checklist to report, for example, "empty work" if the repository does not contain the student's or group's work in the `src` folder of the `develop` branch, or "cheat" in the case of cheating, or if the student or group is unable to explain their work at any time during the review, or if any of the items below are not met.

However, except in cases of cheating, you are encouraged to continue reviewing the project to identify the problems that caused the situation so that they can be avoided in the next review.

Double check that the GIT repository is the one corresponding to the student or group.

Carefully check that nothing malicious has been used to fool you.

In controversial cases, remember that the checklist determines only the general order of the check. The final decision on project evaluation remains with the reviewer.

## Main Part

## Main Part. Task 1

### 1.a. Verification examples

The student should provide examples of applying machine learning methods in real life and explain the benefit of using ML methods in each example.

Score scale in the original checklist: `0 1 2 3 4 5`.

### 1.b. Classification of tasks

Possible classification from the checklist:

**Supervised learning: classification**

- `2`: predict whether a client returns a loan.
- `4`: choose what medicine out of available a patient should take, as multiclass classification.
- `5`: choose segment of clients for a promo communication, with target such as whether the customer participates in promotion or whether the customer buys products from promotion.
- `6`: recognition of defective products, with target `defect` or `not defect`.
- `8`: search sites for input text query, with target such as whether a user opens a link. Multiclass is a bad option here because the number of pages is too big.
- `10`: detect anomaly in site traffic, if we have detected facts of malfunctions.

**Supervised learning: regression**

- `1`: predict house price.
- `3`: predict number of hours from last medication.
- `7`: decide how to place products on a shelf in a store.

**Unsupervised learning: clustering**

- `5`: task is to find users that will be in the same cluster.

**Unsupervised learning: association**

- `5`: task is to find users who are similar to others who participate in or react to a promotion.

**Unsupervised learning: dimensionality reduction**

- `5` and `9`: can be used together with clustering or other approaches.

This division is only an example and does not cover all possible cases.

Score scale in the original checklist: `0 1 2 3 4 5`.

### 1.c. Multiclass and multilabel

Multiclass classification is a classification task with more than two classes; for example, classifying a set of images of fruits that can be oranges, apples, or pears.

Multiclass classification assumes that each sample is assigned to one and only one label: a fruit can be either an apple or a pear, but not both at the same time.

Multilabel classification assigns a set of target labels to each sample. This can be thought of as predicting properties of a data point that are not mutually exclusive, such as topics that are relevant to a document. A text might be about religion, politics, finance, or education, all at the same time, or none of them.

Taken from: <http://scikit-learn.org/stable/modules/multiclass.html>

Score scale in the original checklist: `0 1 2 3 4 5`.

### 1.d. House prices example

The theoretical example is a regression problem.

We can reduce a regression problem to a classification problem by stratifying the target. In other words, it is possible to divide a continuous target into discrete groups.

Score scale in the original checklist: `0 1 2 3 4 5`.

## Main Part. Task 2

### 2.c. Dataset size

Expected dataset shape:

```text
(49352, 15)
```

### 2.d. Target column

Target is `price`.

### 2.e. Empty columns

There are no empty columns.

Score scale in the original checklist: `0 1 2 3 4 5`.

## Main Part. Task 3

### 3.b.i. Target histogram

All values are in 0? There are some large values. We need to find outliers.

Reference image: `materials/etalons/3bi.png`.

### 3.b.ii. Target boxplot

Boxplot shows there are outliers.

Reference image: `materials/etalons/3bii.png`.

### 3.b.iv. Histogram after removing outliers

Removing the outliers allows to see a good distribution.

You need to remove outliers when working with linear regression.

Reference image: `materials/etalons/3biv.png`.

### 3.c.i. Type of `interest_level`

Object type.

### 3.c.ii. Values in `interest_level`

Three values. It is necessary to decode:

```text
low       33697
medium    11116
high       3566
```

### 3.c.iv. Histograms for `bathrooms` and `bedrooms`

There are no outliers for bathrooms and bedrooms.

Reference images:

- `materials/etalons/3civ-1.png`
- `materials/etalons/3civ-2.png`

### 3.d.i. Correlation matrix and heatmap

The maximum correlation is with `bathrooms`.

Reference image: `materials/etalons/3di.png`.

### 3.d.ii. Scatterplots

Reference images:

- `materials/etalons/3dii-1.png`
- `materials/etalons/3dii-2.png`

Score scale in the original checklist: `0 1 2 3 4 5`.

## Main Part. Task 4

### 4.a. Squared features

There are no greater correlations with squared features.

Reference image: `materials/etalons/4a.png`.

Score scale in the original checklist: `0 1 2 3 4 5`.

## Main Part. Task 5

### 5.e. Metrics

Peer's metrics may be somewhat different. It depends on the sklearn version.

Below in the original checklist you can see approximately right metrics.

Reference image: `materials/etalons/5e.png`.

Score scale in the original checklist: `0 1 2 3 4 5`.

## Fails

No specific fail items were extracted beyond the standard review flags described above.

## Record of the Online Review

The original checklist contains an upload area for recording the online review.

Allowed file types:

- `.avi`
- `.mov`
- `.mp4`
- `.webm`

Maximum file size: `800 MiB`.

## Comment

The original checklist contains a comment field for the reviewer.
