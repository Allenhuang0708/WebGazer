import mat from './mat';
import tf from '@tensorflow/tfjs';
import {deep_learning_model} from './index';
const util = {};

/**
 * Eye class, represents an eye patch detected in the video stream
 * @param {ImageData} patch - the image data corresponding to an eye
 * @param {Number} imagex - x-axis offset from the top-left corner of the video canvas
 * @param {Number} imagey - y-axis offset from the top-left corner of the video canvas
 * @param {Number} width  - width of the eye patch
 * @param {Number} height - height of the eye patch
 */

// 224 * 224 for deep learning model
var modelWidth = 224;
var modelHeight = 224;
var resizeWidth = 10;
var resizeHeight = 6;

util.Eye = function(patch, imagex, imagey, width, height) {
    this.patch = patch;
    this.imagex = imagex;
    this.imagey = imagey;
    this.width = width;
    this.height = height;
};

util.getEyeFeats = function(eyes) {
    // face model
    var resizedFace = this.resizeEye(eyes.face, modelWidth, modelHeight);
    resizedFace = this.convertPixels(resizedFace.data, 224, 224);
    var face_tensor = tf.tensor4d(resizedFace);
    var face_output_0 =  deep_learning_model.face[0].predict(face_tensor);
    var [face_input_0, face_input_1] = tf.split(face_output_0, 2, 3);
    var face_output_1_a = deep_learning_model.face[1].predict(face_input_0);
    var face_output_1_b = deep_learning_model.face[2].predict(face_input_1);
    var face_input_2 = tf.concat([face_output_1_a, face_output_1_b], 3);
    var face_output_2_1 = deep_learning_model.face[3].predict(face_input_2);

    //eyes model
    var resizedLeft = this.resizeEye(eyes.left, modelWidth, modelHeight);
    resizedLeft = this.convertPixels(resizedLeft.data, 224, 224);
    var face_tensor = tf.tensor4d(resizedLeft);
    var face_output_0 =  deep_learning_model.eyes[0].predict(face_tensor);
    var [face_input_0, face_input_1] = tf.split(face_output_0, 2, 3);
    var face_output_1_a = deep_learning_model.eyes[1].predict(face_input_0);
    var face_output_1_b = deep_learning_model.eyes[2].predict(face_input_1);
    var face_input_2 = tf.concat([face_output_1_a, face_output_1_b], 3);
    var face_output_2_2 = deep_learning_model.eyes[3].predict(face_input_2);
    
    var resizedRight = this.resizeEye(eyes.right, modelWidth, modelHeight);
    resizedRight = this.convertPixels(resizedRight.data, 224, 224);
    var face_tensor = tf.tensor4d(resizedRight);
    var face_output_0 =  deep_learning_model.eyes[0].predict(face_tensor);
    var [face_input_0, face_input_1] = tf.split(face_output_0, 2, 3);
    var face_output_1_a = deep_learning_model.eyes[1].predict(face_input_0);
    var face_output_1_b = deep_learning_model.eyes[2].predict(face_input_1);
    var face_input_2 = tf.concat([face_output_1_a, face_output_1_b], 3);
    var face_output_2_3 = deep_learning_model.eyes[3].predict(face_input_2);

    var eye_input = tf.concat([face_output_2_2.reshape([-1]), face_output_2_3.reshape([-1])]);
    var eye_output = deep_learning_model.eyes[4].predict(eye_input.reshape([1, -1]));

    var faceGrid = tf.tensor1d(eyes.faceGrid);
    var face_grid = deep_learning_model.face_grid.predict(faceGrid.reshape([1, -1]));

    var connect_input = tf.concat([face_output_2_1, eye_output, face_grid], 1)
    var full_connect = deep_learning_model.full_connect.predict(connect_input);
    var result = full_connect.arraySync();

    return result[0];
}

//Data Window class
//operates like an array but 'wraps' data around to keep the array at a fixed windowSize
/**
 * DataWindow class - Operates like an array, but 'wraps' data around to keep the array at a fixed windowSize
 * @param {Number} windowSize - defines the maximum size of the window
 * @param {Array} data - optional data to seed the DataWindow with
 **/
util.DataWindow = function(windowSize, data) {
    this.data = [];
    this.windowSize = windowSize;
    this.index = 0;
    this.length = 0;
    if(data){
        this.data = data.slice(data.length-windowSize,data.length);
        this.length = this.data.length;
    }
};

/**
 * [push description]
 * @param  {*} entry - item to be inserted. It either grows the DataWindow or replaces the oldest item
 * @return {DataWindow} this
 */
util.DataWindow.prototype.push = function(entry) {
    if (this.data.length < this.windowSize) {
        this.data.push(entry);
        this.length = this.data.length;
        return this;
    }

    //replace oldest entry by wrapping around the DataWindow
    this.data[this.index] = entry;
    this.index = (this.index + 1) % this.windowSize;
    return this;
};

/**
 * Get the element at the ind position by wrapping around the DataWindow
 * @param  {Number} ind index of desired entry
 * @return {*}
 */
util.DataWindow.prototype.get = function(ind) {
    return this.data[this.getTrueIndex(ind)];
};

/**
 * Gets the true this.data array index given an index for a desired element
 * @param {Number} ind - index of desired entry
 * @return {Number} index of desired entry in this.data
 */
util.DataWindow.prototype.getTrueIndex = function(ind) {
    if (this.data.length < this.windowSize) {
        return ind;
    } else {
        //wrap around ind so that we can traverse from oldest to newest
        return (ind + this.index) % this.windowSize;
    }
};

/**
 * Append all the contents of data
 * @param {Array} data - to be inserted
 */
util.DataWindow.prototype.addAll = function(data) {
    for (var i = 0; i < data.length; i++) {
        this.push(data[i]);
    }
};


//Helper functions
/**
 * Grayscales an image patch. Can be used for the whole canvas, detected face, detected eye, etc.
 *
 * Code from tracking.js by Eduardo Lundgren, et al.
 * https://github.com/eduardolundgren/tracking.js/blob/master/src/tracking.js
 *
 * Software License Agreement (BSD License) Copyright (c) 2014, Eduardo A. Lundgren Melo. All rights reserved.
 * Redistribution and use of this software in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 * The name of Eduardo A. Lundgren Melo may not be used to endorse or promote products derived from this software without specific prior written permission of Eduardo A. Lundgren Melo.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @param  {Array} pixels - image data to be grayscaled
 * @param  {Number} width  - width of image data to be grayscaled
 * @param  {Number} height - height of image data to be grayscaled
 * @return {Array} grayscaledImage
 */
util.grayscale = function(pixels, width, height){
    var gray = new Uint8ClampedArray(pixels.length >> 2);
    var p = 0;
    var w = 0;
    for (var i = 0; i < height; i++) {
        for (var j = 0; j < width; j++) {
            var value = pixels[w] * 0.299 + pixels[w + 1] * 0.587 + pixels[w + 2] * 0.114;
            gray[p++] = value;

            w += 4;
        }
    }
    return gray;
};

/* 
Reform the pixel data to: height * width * channel
*/
util.convertPixels = function(pixels, width, height) {
    var matrix = new Array(height);
    for (var row = 0; row < matrix.length; row++) {
        var colArr = new Array(width);    
        for (var col = 0; col < colArr.length; col++) {
            var channelArr = new Array(3);
            for (var channel = 0; channel < channelArr.length; channel++) {
                channelArr[channel] = pixels[row * width * 4 + col * 4 + channel]
            }
            colArr[col] = channelArr;
        }
        matrix[row] = colArr;
    }
    matrix = [matrix]; // make it a 4D
    return matrix;
}

/**
 * Increase contrast of an image.
 *
 * Code from Martin Tschirsich, Copyright (c) 2012.
 * https://github.com/mtschirs/js-objectdetect/blob/gh-pages/js/objectdetect.js
 *
 * @param {Array} src - grayscale integer array
 * @param {Number} step - sampling rate, control performance
 * @param {Array} dst - array to hold the resulting image
 */
util.equalizeHistogram = function(src, step, dst) {
    var srcLength = src.length;
    if (!dst) dst = src;
    if (!step) step = 5;

    // Compute histogram and histogram sum:
    var hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0];

    for (var i = 0; i < srcLength; i += step) {
        ++hist[src[i]];
    }

    // Compute integral histogram:
    var norm = 255 * step / srcLength,
        prev = 0;
    for (var i = 0; i < 256; ++i) {
        var h = hist[i];
        prev = h += prev;
        hist[i] = h * norm; // For non-integer src: ~~(h * norm + 0.5);
    }

    // Equalize image:
    for (var i = 0; i < srcLength; ++i) {
        dst[i] = hist[src[i]];
    }
    return dst;
};

util.equalizeHistogramTest = function(src, step, dst) {
    var srcLength = src.length;
    if (!dst) dst = src;
    if (!step) step = 5;
    step = Math.floor(srcLength / 12);
    var index = 0;
    var totalPixel = 0;
    for (var i = 0; i < srcLength; i += step){
        var segmentPixel = 0;
        for (var j = 0; j < step; j++){
            segmentPixel += src[i + j];
        }
        dst[index] = segmentPixel;
        index += 1;
        totalPixel += segmentPixel;
    }
    for (var i = 0; i < dst.length; i++){
        dst[i] /= totalPixel;
    }
    return dst;
};

util.threshold = function(data, threshold) {
    for (let i = 0; i < data.length; i++) {
        data[i] = (data[i] > threshold) ? 255 : 0;
    }
    return data;
};

util.correlation = function(data1, data2) {
    const length = Math.min(data1.length, data2.length);
    let count = 0;
    for (let i = 0; i < length; i++) {
        if (data1[i] === data2[i]) {
            count++;
        }
    }
    return count / Math.max(data1.length, data2.length);
};

/**
 * Gets an Eye object and resizes it to the desired resolution
 * @param  {webgazer.util.Eye} eye - patch to be resized
 * @param  {Number} resizeWidth - desired width
 * @param  {Number} resizeHeight - desired height
 * @return {webgazer.util.Eye} resized eye patch
 */
util.resizeEye = function(eye, resizeWidth, resizeHeight) {

    var canvas = document.createElement('canvas');
    canvas.width = eye.width;
    canvas.height = eye.height;

    canvas.getContext('2d').putImageData(eye.patch,0,0);

    var tempCanvas = document.createElement('canvas');

    tempCanvas.width = resizeWidth;
    tempCanvas.height = resizeHeight;

    // save the canvas into temp canvas
    tempCanvas.getContext('2d').drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, resizeWidth, resizeHeight);

    return tempCanvas.getContext('2d').getImageData(0, 0, resizeWidth, resizeHeight);
};

/**
 * Checks if the prediction is within the boundaries of the viewport and constrains it
 * @param  {Array} prediction [x,y] - predicted gaze coordinates
 * @return {Array} constrained coordinates
 */
util.bound = function(prediction){
    if(prediction.x < 0)
        prediction.x = 0;
    if(prediction.y < 0)
        prediction.y = 0;
    var w = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
    var h = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
    if(prediction.x > w){
        prediction.x = w;
    }

    if(prediction.y > h)
    {
        prediction.y = h;
    }
    return prediction;
};

/**
 * Write statistics in debug paragraph panel
 * @param {HTMLElement} para - The <p> tag where write data
 * @param {Object} stats - The stats data to output
 */
function debugBoxWrite(para, stats) {
    var str = '';
    for (var key in stats) {
        str += key + ': ' + stats[key] + '\n';
    }
    para.innerText = str;
}

/**
 * Constructor of DebugBox object,
 * it insert an paragraph inside a div to the body, in view to display debug data
 * @param {Number} interval - The log interval
 * @constructor
 */
util.DebugBox = function(interval) {
    this.para = document.createElement('p');
    this.div = document.createElement('div');
    this.div.appendChild(this.para);
    document.body.appendChild(this.div);

    this.buttons = {};
    this.canvas = {};
    this.stats = {};
    var updateInterval = interval || 300;
    (function(localThis) {
        setInterval(function() {
            debugBoxWrite(localThis.para, localThis.stats);
        }, updateInterval);
    }(this));
};

/**
 * Add stat data for log
 * @param {String} key - The data key
 * @param {*} value - The value
 */
util.DebugBox.prototype.set = function(key, value) {
    this.stats[key] = value;
};

/**
 * Initialize stats in case where key does not exist, else
 * increment value for key
 * @param {String} key - The key to process
 * @param {Number} incBy - Value to increment for given key (default: 1)
 * @param {Number} init - Initial value in case where key does not exist (default: 0)
 */
util.DebugBox.prototype.inc = function(key, incBy, init) {
    if (!this.stats[key]) {
        this.stats[key] = init || 0;
    }
    this.stats[key] += incBy || 1;
};

/**
 * Create a button and register the given function to the button click event
 * @param {String} name - The button name to link
 * @param {Function} func - The onClick callback
 */
util.DebugBox.prototype.addButton = function(name, func) {
    if (!this.buttons[name]) {
        this.buttons[name] = document.createElement('button');
        this.div.appendChild(this.buttons[name]);
    }
    var button = this.buttons[name];
    this.buttons[name] = button;
    button.addEventListener('click', func);
    button.innerText = name;
};

/**
 * Search for a canvas elemenet with name, or create on if not exist.
 * Then send the canvas element as callback parameter.
 * @param {String} name - The canvas name to send/create
 * @param {Function} func - The callback function where send canvas
 */
util.DebugBox.prototype.show = function(name, func) {
    if (!this.canvas[name]) {
        this.canvas[name] = document.createElement('canvas');
        this.div.appendChild(this.canvas[name]);
    }
    var canvas = this.canvas[name];
    canvas.getContext('2d').clearRect(0,0, canvas.width, canvas.height);
    func(canvas);
};

/**
 * Kalman Filter constructor
 * Kalman filters work by reducing the amount of noise in a models.
 * https://blog.cordiner.net/2011/05/03/object-tracking-using-a-kalman-filter-matlab/
 *
 * @param {Array.<Array.<Number>>} F - transition matrix
 * @param {Array.<Array.<Number>>} Q - process noise matrix
 * @param {Array.<Array.<Number>>} H - maps between measurement vector and noise matrix
 * @param {Array.<Array.<Number>>} R - defines measurement error of the device
 * @param {Array} P_initial - the initial state
 * @param {Array} X_initial - the initial state of the device
 */
util.KalmanFilter = function(F, H, Q, R, P_initial, X_initial) {
    this.F = F; // State transition matrix
    this.Q = Q; // Process noise matrix
    this.H = H; // Transformation matrix
    this.R = R; // Measurement Noise
    this.P = P_initial; //Initial covariance matrix
    this.X = X_initial; //Initial guess of measurement
};

/**
 * Get Kalman next filtered value and update the internal state
 * @param {Array} z - the new measurement
 * @return {Array}
 */
util.KalmanFilter.prototype.update = function(z) {

    // Here, we define all the different matrix operations we will need
    var add = numeric.add, sub = numeric.sub, inv = numeric.inv, identity = numeric.identity;
    var mult = mat.mult, transpose = mat.transpose;
    //TODO cache variables like the transpose of H

    // prediction: X = F * X  |  P = F * P * F' + Q
    var X_p = mult(this.F, this.X); //Update state vector
    var P_p = add(mult(mult(this.F,this.P), transpose(this.F)), this.Q); //Predicted covaraince

    //Calculate the update values
    var y = sub(z, mult(this.H, X_p)); // This is the measurement error (between what we expect and the actual value)
    var S = add(mult(mult(this.H, P_p), transpose(this.H)), this.R); //This is the residual covariance (the error in the covariance)

    // kalman multiplier: K = P * H' * (H * P * H' + R)^-1
    var K = mult(P_p, mult(transpose(this.H), inv(S))); //This is the Optimal Kalman Gain

    //We need to change Y into it's column vector form
    for(var i = 0; i < y.length; i++){
        y[i] = [y[i]];
    }

    //Now we correct the internal values of the model
    // correction: X = X + K * (m - H * X)  |  P = (I - K * H) * P
    this.X = add(X_p, mult(K, y));
    this.P = mult(sub(identity(K.length), mult(K,this.H)), P_p);
    return transpose(mult(this.H, this.X))[0]; //Transforms the predicted state back into it's measurement form
};

util.getFace = function(imageCanvas, positions) {
    var faceY = Math.round(Math.min(positions[109][1], positions[10][1], positions[338][1]));
    var faceX = Math.round(Math.min(positions[127][0], positions[234][0], positions[93][0]));
    var width = Math.round(Math.max(positions[356][0], positions[454][0], positions[323][0]) - faceX);
    var height = Math.round(Math.max(positions[148][1], positions[152][1], positions[377][1]) - faceY);
    var patch =  imageCanvas.getContext('2d').getImageData(faceX, faceY, width, height);

    var face = {};
    face = {
        patch: patch,
        width: width,
        height: height
    }
    return face;
}

util.getGrid = function(imageCanvas, positions) {
    let width = imageCanvas.width;
    let height = imageCanvas.height;
    let array = [];
    let meshSize = 25;

    const rectWidth = width / meshSize;
    const rectHeight = height / meshSize;
    const xMin = positions[54][0]
    const xMax = positions[365][0]
    const yMin = positions[54][1]
    const yMax = positions[365][1]

    const startPoint1 = Math.ceil(xMin / rectWidth)
    const endPoint1 = Math.ceil(xMax / rectWidth)
    const startPoint2 = Math.ceil(yMin / rectHeight)
    const endPoint2 = Math.ceil(yMax / rectHeight)

    for (let i = 0; i < meshSize; i++) {
        if (i >= startPoint2 && i <= endPoint2) {
            for (let j = 0; j < meshSize; j++) {
                if (j >= startPoint1 && j <= endPoint1) {
                    array.push(1);
                } else {
                    array.push(0);
                }
            }
        } else {
            for (let k = 0; k < meshSize; k++) {
                array.push(0);
            }
        }
    }
    return array;
}
export default util;
