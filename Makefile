################################################################################
#
#   Makefile for DataDrivenEstimator
#
################################################################################

test-cnn:
ifneq ($(OS),Windows_NT)
	mkdir -p testing/coverage
	rm -rf testing/coverage/*
endif
	nosetests --nocapture --nologcapture --all-modules -A 'not helper' --verbose --with-coverage --cover-inclusive --cover-package=dde --cover-erase --cover-html --cover-html-dir=testing/coverage --exe dde
