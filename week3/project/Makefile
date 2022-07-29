CONTAINER_NAME=news-classifier-container

# build the news-classifier image
build:
	docker build --platform linux/amd64 -t news-classifier .

# start a named docker container
run:
	docker run --name $(CONTAINER_NAME) --rm -p 80:80 news-classifier 

# enter the news-classifier container
exec:
	docker exec -it $(CONTAINER_NAME) /bin/sh

# stop the container (notice we use the name of the container)
stop:
	docker stop $(CONTAINER_NAME)

# remove logs.out file (useful before starting to build container) and
# you didn't use .dockerignore to ignore logs.out file if one exists.
clean:
	rm data/logs.out
