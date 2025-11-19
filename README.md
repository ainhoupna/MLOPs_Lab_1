[![CI](https://github.com/ainhoupna/MLOPs_Lab_1/actions/workflows/CI.yml/badge.svg)](https://github.com/ainhoupna/MLOPs_Lab_1/actions/workflows/CI.yml)


# MLOps-Lab1: Initial Image Classification Pipeline

This repository contains the foundational structure for a Deep Learning Image Classification project, developed as part of an MLOps assignment. This initial lab focuses on establishing a robust **Continuous Integration (CI) pipeline** using GitHub Actions, supported by automated testing, linting, and code formatting.

## Project Functionality (Lab 1)

The application provides a module for basic image preprocessing and a simulated prediction mechanism.

| Component | Functionality | Implementation |
| :--- | :--- | :--- |
| **Logic (`mylib/`)** | Image Resizing, Grayscale Conversion, Rotation, and **Randomized Class Prediction** (to be replaced by a real model in subsequent labs). | Python (PIL/Pillow) |
| **CLI (`cli/`)** | Command-line interface to execute core functions (`predict`, `resize`). | Click |
| **API (`api/`)** | RESTful microservice exposing endpoints for prediction and transformations (`/resize`, `/grayscale`, `/rotate`). | FastAPI |

## Development and CI Setup

The entire CI process is managed via the **`Makefile`**, which defines targets for consistent local and remote execution.

### Local Development Commands

Ensure your virtual environment is active before running any commands.

| Command | Action | Tools Used |
| :--- | :--- | :--- |
| `make install` | Installs all project and development dependencies. | `uv` |
| `make format` | Automatically formats all Python code. | `black` |
| `make lint` | Checks code quality and style compliance. | `pylint` |
| `make test` | **Runs all unit and integration tests.** | `pytest`, `pytest-cov` |
| `make all` | Runs `install`, `format`, `lint`, and `test` in sequence. | N/A |

### CI Pipeline (`.github/workflows/ci.yml`)

The CI pipeline runs the `make install`, `make format`, `make lint`, and `make test` targets on every push and pull request to ensure that only tested and styled code is merged into the `main` branch.
