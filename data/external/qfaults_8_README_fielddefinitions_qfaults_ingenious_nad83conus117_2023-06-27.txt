Field definitions for columns in qfaults_ingenious_nad83conus117_2023-06-27.shp

FID
Feature identifier ascribed by GIS software.

Shape
Feature geometry ascribed by GIS software.

NAME
Name of the geologic structure, includes section name and other modifiers.

NUM
Unique number for each fault or fold. A letter is added to the end of the number for faults that have clearly defined sections.

DIPDIRECT
Dip direction for the ENTIRE FAULT OR FAULT SECTION, in cases where a single structure is known to have back thrusts and forethrusts, the dip direction is specific to the line segment record.

SLIPDIRECT
Cardinal direction of slip.

SLIPSENSE
Type of relative fault or fold movement.
Value	Definition
NULL	empty value, slip-sense is not applicable to the structure
unk	slip-sense is unknown
N	normal
R	reverse
SS	strike slip
T	thrust

SECONDARY
Secondary slip.
Value	Definition
NULL	value is unknown or not applicable
SS	strike-slip
T	thrust
R	reverse

SLIPRT2023
The assigned slip rate in mm/year.

SLIPRTNUM
Numeric portion of SLIPRT2023.

REC2023
The upper bounding time (years) of the most recent surface-deforming earthquake.

RECNUM
Numeric portion of REC2023.

CODE2023
A three-integer field that defines certainty or reliability of field mapping.
Consists of: certainty or reliability of field mapping (FCODE, integer one), time of most recent movement (RCODE, integer two), and amount or rate of slip (SLIPCODE, integer three). CODE is used for setting line type (fault trace) to be plotted.

FCODE2023
The first integer in CODE2023; defines how well the fault is located and expressed in the landscape.
Value	Definition
1	fault landforms are more continuous than discontinuous and mapping is accurate at given "MAPSCALE" value
2	fault landforms are more discontinuous than continuous and mapping is accurate at given "MAPSCALE" value
3	location of fault is inferred
4	undefined

RCODE2023
The second integer in CODE2023; defines the upper bounding time of the most recent surface-deforming earthquake.
Value	Definition
1	historic (0 <= RECNUM <= 150)
2	post glacial (150 < RECNUM <= 15,000)
3	late Quaternary (15,000 < RECNUM <= 130,000)
4	middle and late Quaternary (130,000 < RECNUM <= 750,000)
5	Quaternary (750,000 < RECNUM <= 1,600,000)
6	Tertiary (select faults included for coverage; RECNUM > 1,600,000)

SCODE2023
SCODE is the third integer in CODE2023 and defines the assigned slip rate category.
Value	Definition
1	SLIPRTNUM > 5
2	1 < SLIPRTNUM <= 5
3	0.2 < SLIPRTNUM <= 1
4	0 < SLIPRTNUM <= 0.2

FTYPE_
Defines how well the fault is located and expressed in the landscape.
Value			Definition
Well Constrained	fault landforms are more continuous than discontinuous and mapping is accurate at given "MAPPEDSCALE" value
Moderately Constrained	fault landforms are more discontinuous than continuous and mapping is accurate at given "MAPPEDSCALE" value
Inferred		location of fault is inferred

MAPSCALE
Mapped scale can control visualization of the fault at various scales.
Value	Definition
24	1:24,000, fault should be more continuous than discontinuous and mapping is accurate at >10,000 scale.
63	1:63,360, fault should be more continuous than discontinuous and mapping is accurate at >24,000 scale.
100	1:100,000, fault could be more discontinuous than continuous and mapping is accurate at >50,000 scale.
250	1:250,000, fault could be more discontinuous than continuous and mapping is accurate at >1:125,000 scale.
316	1:316,000, fault could be more discontinuous than continuous and mapping is accurate at >1:158,000 scale.
500	1:500,000, fault could be more discontinuous than continuous and mapping is accurate at >1:250,000 scale.

SLIPINFO
Slip rate information source.

RECINFO
Recency information source.

COMMENTS
General comments.

GEOCOMM
Comments on fault geometry.

Shape_Length
Line length ascribed by GIS software (meters).
