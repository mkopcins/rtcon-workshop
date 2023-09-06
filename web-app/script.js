const canvas = new fabric.Canvas('canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 30;
canvas.freeDrawingBrush.color = "#000000";
canvas.backgroundColor = "#ffffff";
canvas.renderAll();


var ctx = $('#myChart')[0].getContext('2d');
var chart;
chart = new Chart(ctx, {
        type: 'bar',
        data: {
            datasets: [{
                backgroundColor: 'rgba(75,192,192,0.2)',
                borderColor: 'rgba(75,192,192,1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    min: 0,
                    max: 1
                }
            }
        }
    });


$("#clear-canvas").click(function(){
  canvas.clear();
  canvas.backgroundColor = "#ffffff";
  canvas.renderAll();
  $("#status").removeClass();
});


function updateChart(newData, top=5) {
    const joined = newData.labels.map((label, i) => ({ label, data: newData.proba[i] }));
    joined.sort((a, b) => b.data - a.data);

    const sortedLabels = joined.map(x => x.label);
    const sortedData = joined.map(x => x.data);

    sortedLabels.splice(top);
    sortedData.splice(top);

    chart.data.labels = sortedLabels;
    chart.data.datasets[0].data = sortedData;
    chart.update();
}


$("#predict").click(function(){
  const dataURL = canvas.toDataURL('jpg');
  const base64Image = dataURL.split(',')[1];

  $.ajax({
    url: 'http://localhost:8081/api/mnist',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ img_base_64: base64Image }),
    success: function (res) {
      if (res.prediction !== undefined) {
        console.log("Result: " + JSON.stringify(res, null, 2));
        $("#response-text").text("Prediction: " + res.prediction);
        updateChart(res);
      } else {
         console.log('Script Error: ' + JSON.stringify(res, null, 2))
      }
    },
    error: function (xhr, textStatus, error) {
      console.log("POST Error: " + xhr.responseText + ", " + textStatus + ", " + error);
    }
  });

});