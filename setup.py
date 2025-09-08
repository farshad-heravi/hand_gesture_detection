from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hand-gesture-detection",
    version="2.0.0",
    author="Farshad Nozad Heravi",
    author_email="f.n.heravi@gmail.com",
    description="Professional hand gesture detection system with MediaPipe and PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/farshad-heravi/hand_gesture_detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hgd-collect=hand_gesture_detection.scripts.collect_data:main",
            "hgd-train=hand_gesture_detection.scripts.train_model:main",
            "hgd-inference=hand_gesture_detection.scripts.run_inference:main",
            "hgd-evaluate=hand_gesture_detection.scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hand_gesture_detection": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
)
