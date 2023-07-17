from flask import flask, jsonify, request
import csv

all_movies = []

with open('movies.csv') as f:
    reader = csv.reader(f)
    data = list(reader)
    all_movies = data[1:]
    #end of the line hit enter

liked_movies = []
did_not_watch = []
not_liked_movies = []

app = flask(__name__)

@app.route("/get-movie")  #create.user.defined.funtion() 
    def get_movies():
        return jsonify({
            "data": all_movies[0],
            "status": "success"
        })


@app.route("/get-movie")  #create.user.defined.funtion() 
    def get_movies():
        return jsonify({
            "data": liked_movies[0],
            "status": "success"
        })

@app.route("/get-movie")  #create.user.defined.funtion() 
    def get_movies():
        return jsonify({
            "data": did_not_watch[0],
            "status": "success"
        })

@app.route("/get-movie")  #create.user.defined.funtion() 
    def get_movies():
        return jsonify({
            "data": not_liked_movies[0],
            "status": "success"
        })

