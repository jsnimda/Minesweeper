import pytest
from PIL import Image

from synthetic_scenes.augmentation import augment_scene


def example():
    board = {'board_id':'test','style':'microsoft','rows':1,'columns':2,
             'boxes':{'grid':(3,4,24,14)},'cell_boxes':[[(3,4,13,14),(14,4,24,14)]],
             'x_edges':[3,13.5,24],'y_edges':[4,14],'cell_states':[[9,1]]}
    return Image.new('RGB',(30,20),'white'), {'size':(30,20),'boards':[board]}


def test_transform_and_augment():
    image,meta=example()
    result,new=augment_scene(image,meta,resize_factor=.63,blur=.2,jpeg_quality=90)
    assert result.size==(19,13)
    assert new['boards'][0]['boxes']['grid']==pytest.approx((3*19/30,4*13/20,24*19/30,14*13/20))
    assert meta['boards'][0]['x_edges']==[3,13.5,24]


def test_augmentation_brightness_contrast_and_noop():
    image, metadata = example()
    original, unchanged = augment_scene(image, metadata)
    assert original is image
    assert unchanged['boards'][0]['boxes']['grid'] == metadata['boards'][0]['boxes']['grid']
    adjusted, _ = augment_scene(image, metadata, brightness=.5, contrast=1.)
    assert adjusted.getpixel((0, 0)) == (127, 127, 127)


def test_render_scene_rejects_unknown_details_before_rendering():
    from synthetic_scenes.render import render_scene

    with pytest.raises(ValueError, match='Unsupported details'):
        render_scene({}, details=('scenes',))
    with pytest.raises(ValueError, match='collection'):
        render_scene({}, details='scene')
