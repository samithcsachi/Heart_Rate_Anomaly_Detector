import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Heart_Rate_Anomaly_Detector"
AUTHOR_USER_NAME = "samithcsachi"
SRC_REPO = "Heart_Rate_Anomaly_Detector"
AUTHOR_EMAIL = "samith.sachi@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="An end to end project to predict or identify abnormal heart rate patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "SRC"},
    packages=setuptools.find_packages(where="src")

)