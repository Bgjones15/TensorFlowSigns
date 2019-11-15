const socket = io.connect("http://localhost:3000");
var model;
$(document).ready(function () {

    

    const modelURL = 'http://localhost:3000/model';

    const preview = document.getElementById("preview");
    const predictButton = document.getElementById("predict");
    const clearButton = document.getElementById("clear");
    const numberOfFiles = document.getElementById("number-of-files");
    const fileInput = document.getElementById('file');

    const predict = async (modelURL) => {
        console.log("stuck");
        console.log(modelURL);
        if (!model) model = await tf.loadLayersModel(modelURL);
        
    }

    
        

});

function predictImage(image){

    console.log(tf.browser.fromPixels(image));

}

function onFileSelected(event){
    let selectedFile = event.target.files[0];
    let reader = new FileReader();
    let imgtag = $( 'img' );
    let img1 = new Image(37,37);
    console.log(imgtag);
    imgtag.title = selectedFile.name;
    reader.onload = function(event){
        console.log(event.target.result);
        imgtag.attr("src",event.target.result);
        img1 = imgtag;
    };
    reader.readAsDataURL(selectedFile);
    
    predictImage(img1);
}