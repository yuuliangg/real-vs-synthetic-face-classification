# Dataset setup

The image dataset is intentionally **not included** in this public repository.

The original coursework used:

- 100 real face images
- 100 synthetic face images
- 5 additional real images for external testing
- 5 additional synthetic images for external testing

To run the notebook or Python script, place images in this structure:

```text
data/
├── RealFace/
│   ├── image1.jpg
│   └── ...
├── SynFace/
│   ├── image1.jpg
│   └── ...
└── test_new/
    ├── RealFace/
    │   └── ...
    └── SynFace/
        └── ...
```

Supported extensions in the cleaned script are `.jpg`, `.jpeg`, and `.png`.

Only use images that you have permission to store and redistribute. The repository ignores the dataset folders by default so private or licensed images are not accidentally committed.
