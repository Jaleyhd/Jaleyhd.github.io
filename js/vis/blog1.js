$(document).ready(function() {
	createDistribution()
	createHeadTailAnimation()
});


function createDistribution(){

mydata = [{
	"label":"TAIL",
	"count":19
  },
  {
	"label":"HEAD",
	"count":21
  }]

colors=['#ccccaa','aacccc']

var maxHeight= d3.max(mydata,function(d){ return d.count})
var maxWidth= d3.max(mydata,function(d){ return 2})


//Math.max.apply(Math,mydata.array.map(function(o){return o.frequency}))
var totalHeight=400//Total Height
var totalWidth=200//Total Width
var totalDivisions=100//Total divisions in xAxis
var totalTicks=12//Total Tics in xAxis
var margin={
	top : 30,
	right : 30,
	bottom : 40,
	left : 50
}
var graphHeight=totalHeight-margin.top-margin.bottom; 
var graphWidth=totalWidth-margin.left-margin.right;
var animateDuration=700
var animateDelay=50
//var barWidth=35
//var barOffset=5
var yScale=d3.scaleLinear()
	.domain([0,maxHeight])
	.range([0,graphHeight])

var xScale=d3.scaleBand()
	.domain(mydata.map(function(d){return d.label}))
	.range([0,graphWidth])
	.padding(0.1)



myChart=d3.select('#visprobdistr')
	.append('svg')
	.attr('width',graphWidth+margin.left+margin.right)
	.attr('height',graphHeight+margin.top+margin.bottom)
	.style('background','#f4f4f4')
	.append('g')
	.attr('transform','translate('+margin.left+','+(margin.top)+')')
	.selectAll('rect')
		.data(mydata)
		.enter().append('rect')
			.style('fill',function(d,i){
				return colors[i]
			})
			.attr('width',xScale.bandwidth())
			.attr('height',function(d){
				return 0;//yScale(d.frequency);
			})
			.attr('x',function(d,i){
				return xScale(d.label);
			})
			.attr('y',function(d){
				return graphHeight;//-yScale(d.frequency);
			})


var vScale=d3.scaleLinear()
	.domain([0,maxHeight])
	.range([graphHeight,0])

var hScale=d3.scaleBand()
	.domain(mydata.map(function(d){return d.label}))
	.range([0,graphWidth])



var vAxis=d3.axisLeft()
	.scale(vScale)
	.ticks(5)
	.tickPadding(5)


var vGuide=d3.select('svg')
		.append('g')

vAxis(vGuide);
vGuide.attr('transform','translate('+margin.left+','+margin.top+')')
vGuide.selectAll('path')
	.style('fill','none')
	.style('stroke','#000')

vGuide.selectAll('line')
	.style('stoke','#000') 

var vGuide=d3.select('svg')
		.append('g')

var hAxis=d3.axisBottom(mydata.map(function(d){return d.label}))
	.scale(hScale)



var hGuide=d3.select('svg')
		.append('g')
		.style("font-size","15px");

hAxis(hGuide);
hGuide.attr('transform','translate('+margin.left+','+(margin.top+graphHeight)+')')
hGuide.selectAll('path')
	.style('fill','none')
	.style('stroke','#000')


hGuide.selectAll('line')
	.style('stoke','#000') 

myChart.transition()
	.attr('height',function(d){
		return yScale(d.count)
	})
	.attr('y',function(d){
		return graphHeight-yScale(d.count)
	})
	.duration(animateDuration)
	.delay(function(d,i){
		return (i+1)*animateDelay
	})
	.ease(d3.easeElastic)


}


















function createHeadTailAnimation(){

random = d3.randomUniform(0,1),
mydata = [0,1,0,1,1,0,0,1,0,0,1,1,1,0,0]
mydict={0:'TAIL',1:'HEAD'}

colors=['#ccccaa','aacccc']
windowspan=mydata.length

var maxHeight= d3.max(mydata)
var maxWidth= mydata.length


//Math.max.apply(Math,mydata.array.map(function(o){return o.frequency}))
var totalHeight=400//Total Height
var totalWidth=400//Total Width
var totalDivisions=100//Total divisions in xAxis
var totalTicks=12//Total Tics in xAxis
var margin={
	top : 30,
	right : 30,
	bottom : 40,
	left : 50
}
var graphHeight=totalHeight-margin.top-margin.bottom; 
var graphWidth=totalWidth-margin.left-margin.right;
var animateDuration=10000
var animateDelay=500
//var barWidth=35
//var barOffset=5
var yScale=d3.scaleLinear()
	.domain([0,maxHeight])
	.range([0,graphHeight])

var xScale=d3.scaleBand()
	.domain(d3.range(0,windowspan))
	.range([0,parseInt(graphWidth/3)*windowspan])
	.padding(0.1)


mySvg=d3.select('#vistossevent')
	.append('svg')

myChart=mySvg.attr('width',graphWidth+margin.left+margin.right)
	.attr('height',graphHeight+margin.top+margin.bottom)
	.style('background','#f4f4f4')
	.append('g')
	.attr('transform','translate('+margin.left+','+(margin.top)+')')
	.selectAll('g')
		.data(mydata.slice(0,windowspan))
		.enter().append('rect')
			.style('fill',function(d){
				return colors[d]
			})
			.attr('width',xScale.bandwidth())
			.attr('height',function(d){
				return parseInt(yScale(1)/2);
			})
			.attr('x',function(d,i){
				return xScale(i)+totalWidth/2;	
			})
			.attr('y',function(d){
				return graphHeight-parseInt(yScale(1)/2)-parseInt(yScale(d)/2);
			})
			.text('Hi')



myChart.transition()
	.duration(animateDuration)
	.ease(d3.easeLinear)
	.on("start", tick)

function tick() {

  // Push a new data point onto the back.
  mydata.push(parseInt(random()>0.5));

  // Redraw the line.
  d3.select(this)
      .attr("d", myChart)
      .attr("transform", null);

  // Slide it to the left.
  d3.active(this)
      .attr("transform", "translate(" + -3*totalWidth + ",0)")
    .transition()
      .on("start", tick);

  // Pop the old data point off the front.
  //mydata.shift();

}


}
 	