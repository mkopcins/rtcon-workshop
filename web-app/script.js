const canvas = new fabric.Canvas('canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 30;
canvas.freeDrawingBrush.color = "#000000";
canvas.backgroundColor = "#ffffff";
canvas.renderAll();


$("#clear-canvas").click(function(){
  canvas.clear();
  canvas.backgroundColor = "#ffffff";
  canvas.renderAll();
  $("#status").removeClass();
});


$("#predict").click(function(){
  const dataURL = canvas.toDataURL('jpg');
  const base64Image = dataURL.split(',')[1];

  $.ajax({
    url: 'http://localhost:8080/api/mnist',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ img_base_64: base64Image }),
    success: function (res) {
      if (res.prediction !== undefined) {
        console.log("Result: " + JSON.stringify(res, null, 2));
        $("#response-text").text("Prediction: " + res.prediction);
      } else {
         console.log('Script Error: ' + JSON.stringify(res, null, 2))
      }
    },
    error: function (xhr, textStatus, error) {
      console.log("POST Error: " + xhr.responseText + ", " + textStatus + ", " + error);
    }
  });

});