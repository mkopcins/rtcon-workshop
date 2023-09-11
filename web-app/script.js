// CANVAS

const canvas = new fabric.Canvas('canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 30;
canvas.freeDrawingBrush.color = "#000000";
canvas.backgroundColor = "#ffffff";
canvas.renderAll();


$("#clear-canvas").click(function () {
    canvas.clear();
    canvas.backgroundColor = "#ffffff";
    canvas.renderAll();
    $("#status").removeClass();
});


// PREDICTIONS USING REST API CALL

$("#predict").click(function () {
    const dataURL = canvas.toDataURL('jpg');
    const base64Image = dataURL.split(',')[1];

    $.ajax({
        url: 'http://localhost:8081/api/mnist',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({img_base_64: base64Image}),
        success: function (res) {
            if (res.prediction !== undefined) {
                console.log(res);
                $("#response-text").text("Prediction: " + res.prediction);
                updateChart(res);
            } else {
                console.log(res)
            }
        },
        error: function (xhr, textStatus, error) {
            console.log("POST Error: " + xhr.responseText + ", " + textStatus + ", " + error);
        }
    });
});



// PREDICTIONS USING ONNX.JS

tf.setBackend('cpu');
let session; // for storing the model session globally
document.addEventListener('DOMContentLoaded', async (event) => {
    const path = '../model.onnx'; // path or URL to your model file
    session = new onnx.InferenceSession();
    await session.loadModel(path);
});

async function extractImage(height = 28, width = 28) {
    let image = canvas.toCanvasElement();
    image = tf.browser.fromPixels(image);
    image = tf.image.rgbToGrayscale(image);
    image = tf.image.resizeBilinear(image, [height, width]);
    image = image.div(255.0);
    image = tf.onesLike(image).sub(image);
    return image.dataSync();
}
async function predictOnnx(tensor) {
    const input = new onnx.Tensor(tensor, 'float32', [1, 1, 28, 28]);
    const outputMap = await session.run([input]);
    const outputData = Array.from(outputMap.values().next().value.data);
    const outputTensor = tf.tensor(outputData);
    const proba = tf.softmax(outputTensor).dataSync();
    const prediction = proba.indexOf(Math.max(...proba));

    const result = {
        "prediction": prediction,
        "proba": proba,
        "labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    };

    return result;
}

$("#predict-onnx").click(
    async function () {
        const tensor = await extractImage();
        const result = await predictOnnx(tensor);
        $("#response-text").text("Prediction: " + result.prediction);
        updateChart(result);
});



// CHART

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

function updateChart(newData, top = 5) {
    const joined = newData.labels.map((label, i) => ({label, data: newData.proba[i]}));
    joined.sort((a, b) => b.data - a.data);

    const sortedLabels = joined.map(x => x.label);
    const sortedData = joined.map(x => x.data);

    sortedLabels.splice(top);
    sortedData.splice(top);

    chart.data.labels = sortedLabels;
    chart.data.datasets[0].data = sortedData;
    chart.update();
}
