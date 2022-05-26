

var data = [];

const TOP_PADDING = 10;
const RIGHT_PADDING = 10;
const BOTTOM_PADDING = 30;
const LEFT_PADDING = 5;

// Full size of the svg element.
const HEIGHT = 400;
const WIDTH = 1500;

const DURATION = 500; // of transitions
let barPadding, barWidth, xAxisGroup, xScale, yAxisGroup, yScale;
// This is used to select bar colors based on their index.
const colorScale = d3.scaleOrdinal(d3.schemePaired); // 12 colors

// Size that can be used for the bars.
const usableHeight = HEIGHT - TOP_PADDING - BOTTOM_PADDING;
const usableWidth = WIDTH - LEFT_PADDING - RIGHT_PADDING;

var connection = new WebSocket('ws://localhost:4041/');


// This returns a text color to use on a given background color.
function getTextColor(bgColor) {
  // Convert the hex background color to its decimal components.
  const red = parseInt(bgColor.substring(1, 3), 16);
  const green = parseInt(bgColor.substring(3, 5), 16);
  const blue = parseInt(bgColor.substring(5, 7), 16);

  // Compute the "relative luminance".
  const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;

  // Use dark text on light backgrounds and vice versa.
  return luminance > 0.5 ? 'black' : 'white';
}

// This updates the attributes of an SVG text element
// that displays the score for a bar.
function updateText(text) {
  myTransition(text)
    .attr('fill', d => {
      const barColor = colorScale(d.colorIndex);
      return getTextColor(barColor);
    })
    .text(d => d)
    .attr('x', barWidth / 2) // center horizontally in bar
    .attr('y', d => TOP_PADDING + yScale(d) + 20); // just below top
}

function updateXAxis(svg, data) {
  if (!xAxisGroup) {
    // Create an SVG group that will hold the x axis and
    // translate the group to the appropriate position in the SVG.
    xAxisGroup = svg
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${TOP_PADDING + usableHeight-usableHeight/2})`);
    xAxisGroup = myTransition(xAxisGroup);
  }

  // Create a scale that maps fruit names to positions on the x axis.
  const xAxisScale = d3
    .scaleBand()
    .domain(data.map(item => item.name)) // fruit names
    .range([LEFT_PADDING, LEFT_PADDING + usableWidth]);

  // Create and call an axis generator function that renders the xAxis.
  const xAxis = d3.axisBottom(xAxisScale).ticks(data.length);
  xAxis(xAxisGroup);
}

function updateYAxis(svg, data, max) {
  if (!yAxisGroup) {
    // Create an SVG group that will hold the y axis and
    // translate the group to the appropriate position in the SVG.
    yAxisGroup = svg
      .append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${LEFT_PADDING}, ${TOP_PADDING})`);
    yAxisGroup = myTransition(yAxisGroup);
  }

  // Create an array with values from zero to max
  // that will be used as the tick values on the y axis.
  const tickValues = Array.from(Array(max + 1).keys());

  // Create an axis generator function that renders the yAxis.
  const yAxis = d3
    .axisLeft(yScale)
    .tickValues(tickValues)
    .tickFormat(n => n.toFixed(0));

  // Pass the selection for the group to the
  // axis generator function to render it.
  yAxis(yAxisGroup);
  // An equivalent way to do this is yAxisGroup.call(yAxis);
}


// This updates the attributes of an SVG rect element
// that represents a bar.
function updateRect(rect) {
  rect
    // Each fruit will keep the same color as its score changes.
    //.attr('fill', d => colorScale(d.colorIndex))
    .attr('width', barWidth - barPadding * 2)
    .attr('height', d => Math.abs(yScale(d)))
    .attr('x', barPadding)
    .attr('y', d => TOP_PADDING +usableHeight - Math.abs(yScale(d)))
    .attr("fill", d => (d>0)?"rgba(180, 190, 250, 1.0)":"rgba(240, 180, 80, 1.0)" );
}

// This updates the bar chart with random data.
function update() {

  // Calculate padding on sides of bars based on # of bars.
  barPadding = 0;//Math.ceil(30 / data.length);

  // Calculate the width of each bar based on # of bars.
  barWidth = usableWidth / data.length;
  // Create a scale to map data index values to x coordinates.
  // This is a function that takes a value in the "domain"
  // and returns a value in the "range".
  xScale = d3
    .scaleLinear()
    .domain([0, data.length])
    .range([LEFT_PADDING, LEFT_PADDING + usableWidth]);

  // Create a scale to map data score values to y coordinates.
  // The range is flipped to account for
  // the SVG origin being in the upper left corner.
  // Like xScale, this is a function that takes a value in the "domain"
  // and returns a value in the "range".
  // The d3.max function computes the largest data value in a given array
  // where values are computed by the 2nd argument function.
  //const max = d3.max(data, d => d);
  //todo send config via websocket
  yScale = d3.scaleLinear().domain([0, 1.5]).range([0, usableHeight]);

  // Create a D3 selection object that represents the svg element
  // and set the size of the svg element.
  const svg = d3.select('#chart').attr('width', WIDTH).attr('height', HEIGHT);

  // This is the most critical part to understand!
  // You learned about about selections and the general update pattern
  // in the previous section.
  const groups = svg
    .selectAll('.bar')
    .data(data, d => d.name)
    .join(enter => {
      // Create a new SVG group element for each placeholder
      // to represent a new bar.
      // For now the only thing in each group will be a rect element,
      // but later we will add a text element to display the value.
      const groups = enter.append('g').attr('class', 'bar');

      // Create a new SVG rect element for each group.
      groups
        .append('rect')
        .attr('height', 0)
        .attr('y', TOP_PADDING + usableHeight);

      return groups;
    });

  // The join method call above returns a selection that combines
  // the update and enter sub-selections into its update selection.
  // This allows operations needed on elements in both
  // to be performed on the new selection.

  // Translate the groups for each bar to their
  // appropriate x coordinate based on its index.
  groups.attr('transform', (_, i) => `translate(${xScale(i)}, 0)`);

  // Update all the rect elements using their newly associated data.
  updateRect(groups.select('rect'));
}



connection.onmessage = function(event) {
    var newData = JSON.parse(event.data);
    document.getElementById("status").innerHTML = "Connected"
    //resetData(ndx, [yearDim, spendDim, nameDim]);
    data = newData;
    update();
}