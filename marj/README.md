We're not really using autoencoder2d_MRI.py anymore, but leaving it in in case we wanna look at, steal bits of code, etc.

You'll need to look at misc/torchio/data/image.py's custom Image.load() function. Your downloaded version of Torchio will be different by default. This is where I'm resizing the T1/T2 images.
