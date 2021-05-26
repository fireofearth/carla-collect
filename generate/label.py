import enum

class ScenarioIntersectionLabel(object):
    """Labels samples by proximity of vehicle to intersections."""

    # NONE : str
    #     Vehicle is not near any intersections
    NONE = 'NONE'
    # UNCONTROLLED : str
    #     Vehicle is near an uncontrolled intersection
    UNCONTROLLED = 'UNCONTROLLED'
    # CONTROLLED : str
    #     Vehicle is near a controlled intersection
    CONTROLLED = 'CONTROLLED'

class ScenarioSlopeLabel(object):
    """Labels samples by proximity of vehicle to slopes."""

    # NONE : str
    #     Vehicle is not near any intersections
    NONE = 'NONE'
    # SLOPES : str
    #     Vehicle is close or on a sloped road
    SLOPES = 'SLOPES'

class BoundingRegionLabel(object):
    """Labels samples whether they are inside a bounding region.
    Use this to select cars on (a) specific lane(s), intersection(s)"""

    # NONE : str
    #     Vehicle is not inside any bounding region
    NONE = 'NONE'
    # BOUNDED : str
    #     Vehicle is inside a bounding region
    BOUNDED = 'BOUNDED'

class SampleLabelMap(object):
    """Container of sample labels, categorized by different types."""
    
    def __init__(self,
            intersection_type=ScenarioIntersectionLabel.NONE,
            slope_type=ScenarioSlopeLabel.NONE,
            bounding_type=BoundingRegionLabel.NONE,
            slope_pitch=0.0):
        self.intersection_type = intersection_type
        self.slope_type = slope_type
        self.bounding_type = bounding_type
        self.slope_pitch = slope_pitch

class SampleLabelFilter(object):
    """Container for sample label filter."""

    def __init__(self,
            intersection_type=[],
            slope_type=[],
            bounding_type=[]):
        """
        Parameters
        ----------
        intersection_type : list of str
        slope_type : list of str
        """
        self.intersection_type = intersection_type
        self.slope_type = slope_type
        self.bounding_type = bounding_type

    def contains(self, _type, label):
        """Check whether a label of type _type is in the filter.

        Parameters
        ----------
        _type : str
            Label type to lookup.
        label : str
            Label to check for existence in filter.

        Returns
        -------
        bool
        """
        return label in getattr(self, _type, [])

class SegmentationLabel(enum.Enum):
    RoadLine = 6
    Road = 7
    SideWalk = 8
    Vehicle = 10
