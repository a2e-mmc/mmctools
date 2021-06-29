from mmctools.wrf.nudging import write_header, write_surface, write_level

with open('OBS_DOMAIN_surface.test','w') as f:
    write_header(f,datetime='2012-02-07 08:48',lat=35.2700,lon=-113.9500,
                 stationID='KIGM',platform='FM-15 ASOS',elevation=1033.,datatype='NETWORK',
                 is_sound=False,bogus=False,levels=1)
    write_surface(f,height=1033.,temperature=277.150,u=(-4.077,0),v=(2.194,0),rh=35.343,psfc=90063.531)

with open('OBS_DOMAIN_rawinsonde.test','w') as f:
    write_header(f,datetime='2012-02-07 12:00',lat=37.7600,lon=-122.2200,
                 stationID='KOAK',platform='FM-35',elevation=6.,datatype='NETWORK',
                 is_sound=True,bogus=False,levels=3)
    write_level(f,pressure=100800.,height=6.,temperature=284.750,u=(-2.939,0),v=(0.943,0),rh=80.004)
    write_level(f,pressure=100000.,height=70.,temperature=284.950,u=(-3.725,256),v=(2.611,256),rh=(75.761,16384))
    write_level(f,pressure=(98800.,256),height=(170.522,256),temperature=(285.114,256),u=(-4.904,256),v=(5.114,256),rh=(68.341,16640))
