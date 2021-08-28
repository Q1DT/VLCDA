# VLCDA
This is the implementation of paper "Accurately Predicting circRNA-disease Associations Using Variational Graph Auto-encoders and LightGBM".

## Related Files

### Dataset:

```
========================================================================================
| FILE NAME            | DESCRIPTION                                                   |
========================================================================================
|CircR2Disease_circRNA-disease associations.xlsx | known circRNA-disease associations. |
|known disease-circRNA association number ID.xlsx| known circRNA-disease associations. |
|disease_sm.txt        | disease semantic similarity matrix.                           |
|Normal_circRNA_RPM.txt| circRNA expression profile.                                   |
|circRNA number ID.txt | id of circRNA.                                                |
|disease number ID.txt | id of disease.                                                |
|exoRBase-circR2disease id conversion.txt | Unify the format of circRNA id data.       |
```
### Code:

```
========================================================================================
| FILE NAME       | DESCRIPTION                                                        |
========================================================================================
| XGBCDA.py       | function predicting potential circRNA-disease associations.        |
| feature_extract.py  | function extracting feature                                    |
```

