/*
 * This function calculates a measurement for how precise
 * the eye tracker currently is which is displayed to the user
 */
function calculatePrecision(pastArray) {
  const windowHeight = $(window).height();
  const windowWidth = $(window).width();

  // Retrieve the last 50 gaze prediction points
  const x = pastArray[0];
  const y = pastArray[1];

  // Calculate the position of the point the user is staring at
  const staringPointX = windowWidth / 2;
  const staringPointY = windowHeight / 2;

  const precisionPercentages = new Array(x.length);
  calculatePrecisionPercentages(precisionPercentages, windowHeight, x, y, staringPointX, staringPointY);
  const precision = calculateAverage(precisionPercentages);

  // Return the precision measurement as a rounded percentage
  return Math.round(precision);
};

/*
 * Calculate percentage accuracy for each prediction based on distance of
 * the prediction point from the centre point (uses the window height as
 * lower threshold 0%)
 */
function calculatePrecisionPercentages(precisionPercentages, windowHeight, x, y, staringPointX, staringPointY) {
  for (let i = 0; i < precisionPercentages.length; i++) {
    // Calculate distance between each prediction and staring point
    var xDiff = staringPointX - x[i];
    var yDiff = staringPointY - y[i];
    var distance = Math.sqrt((xDiff * xDiff) + (yDiff * yDiff));

    // Calculate precision percentage
    var halfWindowHeight = windowHeight / 2;
    var precision = 0;
    if (distance <= halfWindowHeight && distance > -1) {
      precision = 100 - (distance / halfWindowHeight * 100);
    } else if (distance > halfWindowHeight) {
      precision = 0;
    } else if (distance > -1) {
      precision = 100;
    }

    // Store the precision
    precisionPercentages[i] = precision;
  }
}

/*
 * Calculates the average of all precision percentages calculated
 */
function calculateAverage(precisionPercentages) {
  let precision = 0;
  for (let i = 0; i < precisionPercentages.length; i++) {
    precision += precisionPercentages[i];
  }
  precision /= precisionPercentages.length;
  return precision;
}
