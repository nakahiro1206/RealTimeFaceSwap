## Overview

This project is real-time face swapping with InsightFace.

For quick processing even by low spec PC, I aim to re-organize insightface and make the class serializable!

## naiive-implementation

./single-thread.py

- See it to understand the entire process.
But each process takes 1.5 sec, it cannot be said real-time.

./Script

- Disassembling insightface module

other python project

- my struggling for multi-processing. 20240614-manager.py is nearest to the goal, still far from it though.

Note:
Due to GIL, Python's unique lock system, threading does not work as I expect.

## Goal
### Face swap class
Implemented flow: model_zoo.get_model("inswapper_128.onnx") returns Inswapper through ModelRouter. Inswapper has session member(class name is "PickableInferenceSession(onnxruntime.InferenceSession)"), and this session is run when face swapping.

Structure is too complex to serialize(= be picklable). So, I would like to simplify the flow. The disirable format is ./Script/PicklableTemplate.py
