from setuptools import setup, find_packages

setup(
    name="cooking_ai_pipeline",
    version="0.1.0",
    description="Multimodal AI pipeline for recipe extraction, vision-based cooking step analysis, and contextual troubleshooting.",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/yourusername/cooking-ai-pipeline",
    packages=find_packages(exclude=["tests", "docs"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "requests",
        "tqdm",
        "PyYAML",
        "sqlalchemy",
        "psycopg2-binary",
        "ultralytics",
        "pytube",
        "youtube-transcript-api",
        "google-api-python-client",
        "transformers",
        "torch",
        "Pillow",
        "loguru"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run-vision=vision_data_collector:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
