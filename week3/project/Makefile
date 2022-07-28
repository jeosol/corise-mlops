# build the news-classifier image
build:
	docker build --platform linux/amd64 -t news-classifier .

# start a named container
run:
	docker run --name news-classifier --rm -p 80:80 news-classifier 

# enter the news-classifier container
exec:
	docker exec -it news-classifer /bin/sh

# remove logs.out file (useful before starting to build container) and
# you didn't use .dockerignore to ignore logs.out file if one exists.
clean:
	rm data/logs.out
