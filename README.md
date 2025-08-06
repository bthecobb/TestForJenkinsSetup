# TestForJenkinsSetup  

**Automated build, test and reporting for a C++ project using Jenkins and GoogleTest (gtest).**  

---  

## Table of Contents  
1. [What This Repository Provides](#what-this-repository-provides)  
2. [Prerequisites](#prerequisites)  
3. [Repository Layout](#repository-layout)  
4. [Jenkinsfile – The Heart of the CI](#jenkinsfile---the-heart-of-the-ci)  
5. [Setting Up the Jenkins Job](#setting-up-the-jenkins-job)  
6. [Triggering Builds Automatically](#triggering-builds-automatically)  
7. [Understanding the Pipeline Stages](#understanding-the-pipeline-stages)  
8. [Viewing Test Results in Jenkins](#viewing-test-results-in-jenkins)  
9. [Optional Extras (Coverage, Static‑Analysis, Notifications)](#optional-extras)  
10. [Troubleshooting & FAQ](#troubleshooting--faq)  
11. [Contributing](#contributing)  
12. [License](#license)  

---  

## What This Repository Provides  

- **A ready‑to‑use `Jenkinsfile`** (Declarative Pipeline) that:  
  - Checks out the latest code on every push.  
  - Configures, builds, and runs a GoogleTest suite.  
  - Emits a JUnit‑compatible XML report (`gtest-report.xml`).  
  - Publishes the report to Jenkins so you get the familiar “Test Result” view.  

- **Documentation** (this README) that walks you through:  
  - Required tools and plugins.  
  - How to create the Jenkins job (Multibranch or classic Pipeline).  
  - How to set up automatic push triggers (GitHub/Bitbucket/GitLab).  
  - How to extend the pipeline for coverage, static analysis, or notifications.  

---  

## Prerequisites  

| Category | What you need | Why |
|----------|----------------|-----|
| **Jenkins** | Jenkins ≥ 2.361 (LTS) on a Linux node (or Docker agent). | Recent versions include Declarative Pipeline support and a better UI. |
| **Plugins** | - **Pipeline** (core) <br> - **Git** (or your SCM plugin) <br> - **JUnit** <br> - **GitHub Branch Source** (if using GitHub) <br> - *Optional:* **Slack**, **Email Extension**, **Coverage**, **Warnings NG** | Required for pipeline syntax, SCM checkout, test result parsing and optional extras. |
| **Build Agent** | A Jenkins node (label e.g. `linux-build`) that has: <br> - C++ compiler (`g++` or `clang++`) <br> - `cmake` (≥ 3.10) <br> - `make` (or the tool you use) <br> - `git` <br> - (Optional) `gcov`/`lcov` for coverage | All commands in the pipeline run on this node. |
| **GoogleTest** | Either: <br> - System package (`libgtest-dev` + compiled static lib) <br> - Or add gtest as a submodule and build it as part of the project | The test binary must link against gtest. |
| **SCM Access** | Read access to the repository (HTTPS/SSH token). | `checkout scm` needs credentials to pull source. |

> **Tip:** If you prefer containers, replace the `agent { label … }` block in the `Jenkinsfile` with a Docker image (e.g., `docker { image 'gcc:13' }`). Make sure the image contains the tools listed above. 
