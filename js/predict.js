const categories = ['Added lane', 'Keep right', 'Lane ends', 'Merge', 'Pedestrian crossing', 'School', 'Signal ahead', 'Stop', 'Yield']
const modelURL = 'https://kollinb.github.io/model/model.json'
let model;

async function loadModel() {
    console.log('Loading model from ' + modelURL)

    model = undefined;
    try {
        model = await tf.loadLayersModel(modelURL)
        model.summary()
    } catch(err) {
        console.log('Failed to load model')
        console.log(err)
    }
}

async function handleFiles(files) {
    await predictImage(files[0])
}

async function predictImage(file) {
    if(model == undefined) {
        await loadModel()
    }

    let reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onloadend = async function () {
        $('.loading').css('display', 'block')
        $('.file-upload-label').css('display', 'none')

        let img = document.createElement('img')
        img.src = reader.result

        // Conversion to grayscale would be mean(2) and then expandDims(2)
        // won't work currently because conv2d input layer expects rgb [1, 37, 37, 3]
        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([37, 37])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(0)

        let predictions = await model.predict(tensor).data()
        let results = Array.from(predictions)
            .map(function (p, i) {
                return {
                    probability: p,
                    className: categories[i]
                };
            }).sort(function (a, b) {
                return b.probability - a.probability;
            }).slice(0, 9);
        
        createResultCard(results, img)

        $('.loading').css('display', 'none')
        $('.file-upload-label').css('display', 'block')
    }
}

function createResultCard(results, img) {
    let resultDiv = $("<div>", { "class": "result" })
    resultDiv.append($('<picture/>').append(img))

    let predictionsDiv = $("<div>", { "class": "predictions" })
    results.forEach(result => {
        let predDiv = $("<div>", { "class": "prediction" })
        let predClass = $("<p>", { "class": "prediction-class" })
        let predResult = $("<p>", { "class": "prediction-percent"})

        predClass.html(result.className)
        predResult.html(result.probability.toFixed(2) + '%')

        predDiv.append(predClass)
        predDiv.append(predResult)
        predictionsDiv.append(predDiv)
    });

    resultDiv.append(predictionsDiv)
    $('.results').append(resultDiv) 
}

$(document).ready(function() {
    //Preload model
    loadModel()

    // Handle drag and drop file upload
    $("#upload-box").on('drag dragstart dragend dragover dragenter dragleave drop', function (e) {
        //Prevent any default drag and drop events from happening
        e.preventDefault()
        e.stopPropagation()
    }).on('dragover dragenter', function (e) {
        $("#upload-box").css("background-color", "lightgray")
    }).on('dragleave dragend drop', function(e) {
        $("#upload-box").css("background-color", "#f9f9f9")
    }).on('drop', function(e) {
        handleFiles(e.originalEvent.dataTransfer.files)
    });

    // Handle manual file upload
    $("#file-upload").on('change', function (e) {
        handleFiles(this.files)
    });
});