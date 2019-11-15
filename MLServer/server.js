const express = require('express');
const socket = require('socket.io');


const fs = require('fs');

var connections = [];

const app = express();
app.use(express.static("public"));

const server = app.listen(3000, function () {
    console.log("Server running...");
});

const io = socket(server);

io.on('connection', function (socket) {
    connections.push(socket);

    console.log(socket.id + " has connected");

    socket.on('disconnect', function (data) {
        if (socket.username) {
            io.sockets.emit('log', socket.username + " has left the chat!");
        }
        connections.splice(connections.indexOf(socket), 1);

    });


});

app.get('/model', function (req, res) {
    let rawdata = fs.readFileSync("model/model.json");
    res.send(JSON.parse(rawdata));
});

app.get('/weight', function (req, res){
    let rawdata = fs.readFileSync("model/group1-shard1of1.bin");
    res.send(rawdata);
});