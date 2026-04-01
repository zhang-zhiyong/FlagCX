build: libsdccl-net-example.so

libsdccl-net-example.so: plugin.cc
	g++ -Isdccl -fPIC -shared -o $@ $^

clean:
	rm -f libsdccl-net-example.so
