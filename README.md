# TradingView-Dev-Sandbox
This repo Demonstrates the study of Learning Models for finanical markets.

# DISCLAIMER
I AM NOT Responsible for any damages, finanical loss, or criminal intent.
This Model may and will generate false signals, This script is NOT Financial Advice. Any of the 'BackTested' Strategies are backtested by me, and only me, based on this, user discretion is advised.

Future Releases:
In the future, changes to the preproccessing, data collection, and possibly converting RSI to Volume. This Model is in it's very early stages, Understand whole revisions of the code and trained models will be done. Enjoy the Journal!


KEY:

CBT - Completed and BackTested

POC - Proof of Concept

ML - Machine Learning

NN - Neural Network


-----------------------------------------------------------------------
# Technical Explaination of MACD Perceptron

This repository contains a Python script that implements a multi-timeframe perceptron trading indicator. The script fetches historical data for a specified symbol and multiple timeframes using the TradingView API. It calculates the Moving Average Convergence Divergence (MACD) and Signal Line for each timeframe and trains a perceptron model to predict the next trading signal based on the MACD difference and closing price.

The script then combines the predictions from the individual models using a voting system to generate a final signal. The combined signal is generated only if there is enough agreement between the models for different timeframes.

The main features of this project include:

Fetching historical data for multiple timeframes using the TradingView API.
Calculating MACD and Signal Line for each timeframe.
Preparing datasets and training a perceptron model for each timeframe.
Implementing a voting system to make a combined prediction based on the trained models.

Requirements:

Python 3.x

numpy

pandas

tradingview_ta

scikit-learn

Usage:

Clone the repository.

Install the required packages.

Modify the symbol variable in the script to the desired stock symbol.

Run the script using python main.py.

The script will output the model accuracy for each timeframe and the combined prediction based on the voting system.
