.PHONY: build run

build:
	docker build . -f Dockerfile -t smaug

run:
	docker run -it --rm \
		--mount source=smaug-workspace,target=/workspace smaug
