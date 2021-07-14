# LoFi Girl AI

Recursive Neural Network to create new compositions of LoFi music.

Currently the model takes MIDI files with the piano melody as input and uses them as training data. This is pretty awful and outputs what is barely music, therefore a couple of tasks must be completed to make this really output decent to slightly mediocre LoFi tracks.

- [ ] Get better piano data to train
- [ ] Make it prioratize basic music theory (e.g. follow a Major Pentatonic Scale)
- [ ] Give it structre (start, end, loop)
- [ ] Add drums 

To achieve this most likely a new model has to be made. 
A new model that:

- [ ] Can translate already recorded music into notes by instrument 
*or at least give a decent approximation*
- [ ] Can get decent tracks to use as input 

and apply this data to train the first model.

It is unknown to me if this will work but if it does you could recreate a crappier version of almost any generic pop song as well.

Tested in a python virtual env (pipenv) run:
```
pipenv install
pipenv shell
```
