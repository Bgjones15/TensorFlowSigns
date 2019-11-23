const categories = ['Added lane', 'Keep right', 'Lane ends', 'Merge', 'Pedestrian crossing', 'School', 'Signal ahead', 'Stop', 'Yield']
const modelURL = 'https://kollinb.github.io/model/model.json'
let model;

async function loadModel() {
    console.log('Loading model from ' + modelURL)

    model = undefined;
    try {
        model = await tf.loadLayersModel(modelURL)
    } catch(err) {
        console.log('Failed to load model')
        console.log(err)
    }
}

async function handleFiles(files) {
    previewImage(files[0])
    let predictionResult = await predictImage(files[0]) // should probably wait for them to press a button in prod
}

async function predictImage(file) {
    if(model == undefined) {
        await loadModel()
    }

    let reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onloadend = async function () {
        let img = document.createElement('img')
        img.src = reader.result
        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([37, 37])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(0)
        tensor.print()

        let predictions = await model.predict(tensor)
        console.log(predictions.values)
        predictions.print()
        predictions = predictions.data()
        let results = Array.from(predictions)
            .map(function (p, i) {
                return {
                    probability: p,
                    className: categories[i]
                };
            }).sort(function (a, b) {
                return b.probability - a.probability;
            }).slice(0, 9);
        console.log(results)
    }
}

function previewImage(file) {
    let reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onloadend = function () {
        let img = document.createElement('img')
        img.id = 'uploaded-img'
        img.src = reader.result
        img.style.height = '200px'
        img.style.width = '200px'
        document.getElementById('result-container').appendChild(img)
    }
}


$(document).ready(function() {
    // Handle drag and drop events
    let dropArea = document.getElementById('upload-box')
    $("#upload-box").on('drag dragstart dragend dragover dragenter dragleave drop', function (e) {
        //Prevent any default drag and drop events from happening
        e.preventDefault()
        e.stopPropagation()
    }).on('dragover dragenter', function (e) {
        //highlight upload box
    }).on('dragleave dragend drop', function(e) {
        //unhighlight upload box
    }).on('drop', function(e) {
        handleFiles(e.originalEvent.dataTransfer.files)
    });

    $("#file-upload").on('change', function (e) {
        handleFiles(this.files)
    });

    //Preload model
    loadModel()
});