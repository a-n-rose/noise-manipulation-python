## Very Basic Selective Noise Filter

This will perform a simple demonstration of reducing or removing a noise in your environment. If you run the script, it will record for 5 seconds. The goal here is to tell the program which noise you don't like. 

Next it will record for 20 seoncds. This is to get the natural environment, however that is. You can be talking, writing, listening to music, whatever.

The program will then save the noise, your environment, as well as a noise reduced version of your environment.

While improvements are needed, for example, postfiltering for removing musical noise, this can offer a basis for further development, for example, a real-time selective noise cancellation tool. 

### Installation

It is expected you have Python3.

Set up your virtual environment (I would suggest).

Once set up:

```
(env)..$ pip install -r requirements.txt
```

### Run

To run the program, simply type into the command line:

```
(env)..$ python3 selective_noise_filter.py
```


