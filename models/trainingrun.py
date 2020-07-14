from mongoengine import Document, DateTimeField, ListField, ReferenceField, DictField, FloatField, IntField, \
    BinaryField, StringField


class Episode(Document):
    """everything per cycle"""
    episode = IntField()
    rewards = ListField(FloatField())
    activation = ListField(FloatField())#stores the average activation of the input layer
    weights = BinaryField()
    weights_human = ListField(ListField(FloatField()))
    neuromodulator = ListField(FloatField())

class Trainingrun(Document):
    time_start = DateTimeField()
    time_end = DateTimeField()
    time_elapsed = FloatField()  # in s
    #from exp import Experiment
    #Reference field causes circular import
    instances = ListField(StringField())
    gridsearch = DictField()
