# 🎬 Lab 1 – Movie Recommendation System

This project implements a **hybrid movie recommendation system** as part of a machine learning lab assignment. The system combines **content-based filtering** and **collaborative filtering** to generate movie recommendations.

---

## 📌 Overview

The goal of this lab is to build a recommendation system using the MovieLens dataset. By combining different recommendation techniques, the system is able to suggest movies based on both:

- Movie features (e.g. genres, tags)
- User preferences (ratings)

---

## 🧠 Method

The system uses a **hybrid approach**, combining:

- **Content-based filtering** using movie genres and tags
- **Collaborative filtering** based on user ratings
- **Cosine similarity** to measure similarity between movies

This allows the system to recommend movies based on both content similarity and user behavior.

---

## ⚙️ Implementation

The implementation follows these steps:

1. Read the instructions in the `README.md` inside the `data-files` directory and download the dataset

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run either `main.py` or `pipeline.py` to set up the data and model  
   (this can also be done using the `model_setup()` function in `pipeline.py`)

4. To generate recommendations:
   - Run `main.py` for a terminal-based recommender  
   - Or use the functions in `pipeline.py` to build your own interface