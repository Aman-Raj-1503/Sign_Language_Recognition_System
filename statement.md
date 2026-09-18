# Project Statement

## Project Title
Sign Language Recognition System

## Course
Computer Vision

## Problem Statement
Communication between hearing/speaking individuals and members of the Deaf and hard-of-hearing community is often hindered by a lack of shared language. Sign language provides a rich, structured visual language, but most people without training in it cannot interpret it. This project addresses that gap by building a Computer Vision system that observes a person's hand through a standard webcam and automatically recognizes which sign (letter or word gesture) is being shown, displaying the result as readable text in real time — without requiring any specialized hardware such as gloves or depth sensors.

## Scope of the Project
- The system recognizes a configurable set of **static hand gestures** (a subset of the ASL alphabet plus a few common words: HELLO, THANKS, YES, NO) from a single hand, captured via a standard 2D RGB webcam.
- The pipeline covers the full lifecycle: **data collection → preprocessing → training → evaluation → real-time inference**, all operable from the command line.
- The scope is deliberately limited to **static, single-frame gestures** (not continuous/dynamic signing with motion, and not full sentence-level sign language grammar), which keeps the project tractable within a course project timeframe while still demonstrating the core Computer Vision and Machine Learning concepts (feature extraction, normalization, classification, and evaluation).
- The system is designed to be **extensible**: adding a new gesture only requires collecting labelled samples for it and retraining — no code changes are required.

## Target Users
- **Individuals learning sign language** who want quick feedback on whether their hand shape matches a target gesture.
- **Educators / demonstrators** who want a lightweight, no-hardware-required tool to demonstrate Computer Vision + ML concepts.
- **Developers** who want a clean, modular starting point for building richer sign-language or gesture-recognition applications.
- **Course evaluators**, who can run the entire pipeline end-to-end from the command line without any GUI-only steps.

## High-Level Features
1. **Webcam-based data collection** of labelled hand-gesture samples, stored as normalized landmark feature vectors in a CSV dataset.
2. **Hand landmark extraction** using MediaPipe Hands (21 3D keypoints per hand).
3. **Translation- and scale-invariant preprocessing**, so recognition is robust to the hand's distance from the camera and its position in the frame.
4. **Configurable classical ML classifier** (Random Forest / SVM / MLP) trained on the collected dataset.
5. **Quantitative evaluation** via a train/test split, producing accuracy, per-class precision/recall/F1, and a confusion-matrix visualization.
6. **Real-time recognition** with temporal smoothing (majority vote across recent frames) and a confidence threshold that suppresses low-confidence guesses, improving perceived reliability.
7. **Single unified command-line interface** (`python -m src.main <collect|train|evaluate|recognize>`) so the whole system is operable without any GUI application, satisfying full CLI-executability.
