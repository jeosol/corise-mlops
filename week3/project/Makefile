# build the news-classifier image
build:
	docker build --platform linux/amd64 -t news-classifier .

# start a named container
run-name:
	docker run --name news-classifier -p 80:80 news-classifier 
# start an unnamed container
run:
	docker run -p 80:80 news-classifier 

# enter the news-classifier container
exec:
	docker exec -it news-classifier /bin/sh

# stop the container (notice we use the name of the container)
stop:
	docker stop news-classifier

# remove logs.out file (useful before starting to build container) and
# you didn't use .dockerignore to ignore logs.out file if one exists.
clean:
	rm data/logs.out
