# [WebGazer.js](https://webgazer.cs.brown.edu)

WebGazer.js is an eye tracking library that uses common webcams to infer the eye-gaze locations of web visitors on a page in real time. The eye tracking model it contains self-calibrates by watching web visitors interact with the web page and trains a mapping between the features of the eye and positions on the screen. WebGazer.js is written entirely in JavaScript and with only a few lines of code can be integrated in any website that wishes to better understand their visitors and transform their user experience. WebGazer.js runs entirely in the client browser, so no video data needs to be sent to a server. WebGazer.js can run only if the user consents in giving access to their webcam.

* [Official website](https://webgazer.cs.brown.edu)
* [API Docs](https://github.com/brownhci/WebGazer/wiki/Top-Level-API)

## Features

* Real time gaze prediction on most major browsers
* No special hardware; WebGazer.js uses your webcam
* Self-calibration from clicks and cursor movements
* Easy to integrate with a few lines of JavaScript
* Swappable components for eye detection
* Multiple gaze prediction models
* Useful video feedback to user

## Build the repository

If you want to build the repository from source follow these instructions:

    # Ensure NodeJS is downloaded: https://nodejs.org/en/download/
    git clone https://github.com/brownhci/WebGazer.git
    cd WebGazer
    #install the dependencies
    npm install
    #build the project
    npm run build

open calibration.html, the model should be working
