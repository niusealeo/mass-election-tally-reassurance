from __future__ import annotations 

import json 
import re 
import math 
from datetime import datetime ,timezone 
from pathlib import Path 
from typing import Any ,Dict ,List ,Optional ,Tuple ,Union 

import pandas as pd 


Number =Union [int ,float ]


def now_stamps ()->Tuple [str ,str ]:
    utc =datetime .now (timezone .utc )
    local =utc .astimezone ()
    utc_s =utc .isoformat (timespec ="seconds")
    local_s =local .strftime ("%a %d %b %Y %H:%M:%S %Z%z")
    return utc_s ,local_s 


def safe_mkdir (p :Path )->None :
    p .mkdir (parents =True ,exist_ok =True )


def _json_coerce_numbers (x :Any )->Any :
    """Convert floats that are mathematically integers into int, recursively."""
    if isinstance (x ,float ):
        if x .is_integer ():
            return int (x )
        return x 
    if isinstance (x ,dict ):
        return {k :_json_coerce_numbers (v )for k ,v in x .items ()}
    if isinstance (x ,list ):
        return [_json_coerce_numbers (v )for v in x ]
    return x 


def write_json (path :Path ,obj :Any )->None :
    obj2 =_json_coerce_numbers (obj )
    path .write_text (json .dumps (obj2 ,ensure_ascii =False ,indent =2 ),encoding ="utf-8")


def _decode_best_effort (raw :bytes )->str :
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try :
            return raw .decode (enc ,errors ="strict")
        except UnicodeDecodeError :
            pass 
    return raw .decode ("utf-8",errors ="replace")


def read_csv_robust (path :Path )->pd .DataFrame :
    raw =path .read_bytes ()
    text =_decode_best_effort (raw )
    from io import StringIO 
    return pd .read_csv (StringIO (text ),engine ="python")


def read_csv_atomic (path :Path )->pd .DataFrame :
    """Try skipping a small metadata preamble and choose the best parse."""
    raw =path .read_bytes ()
    text =_decode_best_effort (raw )
    lines =text .splitlines (True )

    best_df :Optional [pd .DataFrame ]=None 
    best_score =-1 

    from io import StringIO 
    for skip in range (0 ,6 ):
        sio =StringIO ("".join (lines [skip :]))
        try :
            df =pd .read_csv (sio ,engine ="python")
        except Exception :
            continue 
        cols =[str (c ).lower ()for c in df .columns ]
        score =0 
        if any ("total valid"in c for c in cols ):
            score +=10 
        if any ("informal"in c for c in cols ):
            score +=5 
        score +=min (df .shape [1 ],40 )
        if score >best_score :
            best_df =df 
            best_score =score 

    return best_df if best_df is not None else read_csv_robust (path )


def _to_num (s :pd .Series )->pd .Series :
    if s .dtype ==object :
        s2 =s .astype (str ).str .replace (",","",regex =False ).str .strip ()
    else :
        s2 =s 
    return pd .to_numeric (s2 ,errors ="coerce")


def _strip_trailing_zeros (v ):
    """For CSV writing: remove meaningless trailing zeros.

    Examples:
      10.0 -> 10
      10.50 -> 10.5
      0.333300000 -> 0.3333
    """
    if v is None or (isinstance (v ,float )and pd .isna (v )):
        return v 
    if isinstance (v ,int ):
        return v 
    if isinstance (v ,float ):
        if float (v ).is_integer ():
            return int (v )
        return f"{v:.12f}".rstrip ("0").rstrip (".")
    return v 


def _format_df_numbers_for_csv (df :pd .DataFrame )->pd .DataFrame :
    out =df .copy ()
    for c in out .columns :
        if pd .api .types .is_numeric_dtype (out [c ]):
            out [c ]=out [c ].apply (_strip_trailing_zeros )
    return out 


def location_pair (df :pd .DataFrame ,i :int )->List [str ]:
    """A size-2 array from the first two columns. No row index."""
    v0 =df .iloc [i ,0 ]if df .shape [1 ]>=1 else ""
    v1 =df .iloc [i ,1 ]if df .shape [1 ]>=2 else ""
    return [
    ""if pd .isna (v0 )else str (v0 ).strip (),
    ""if pd .isna (v1 )else str (v1 ).strip (),
    ]


def _label_cols (df :pd .DataFrame )->List [str ]:
    if df .shape [1 ]>=2 :
        return [df .columns [1 ],df .columns [0 ]]
    return [df .columns [0 ]]


def find_totals_row (df :pd .DataFrame )->Optional [int ]:
    if df .shape [0 ]==0 :
        return None 
    hits :List [int ]=[]
    for idx in df .index :
        for c in _label_cols (df ):
            v =str (df .loc [idx ,c ]).strip ().lower ()
            if v and v !="nan"and "total"in v :
                hits .append (int (idx ))
                break 
    return hits [-1 ]if hits else None 


def candidate_numeric_cols (df :pd .DataFrame )->Tuple [List [str ],str ,str ]:
    cols =list (df .columns )
    total_valid =next ((c for c in cols if str (c ).strip ().lower ()=="total valid candidate votes"),cols [-2 ])
    informal =next ((c for c in cols if "informal"in str (c ).strip ().lower ()),cols [-1 ])
    idx_total_valid =cols .index (total_valid )
    idx_informal =cols .index (informal )
    end =min (idx_total_valid ,idx_informal )
    cand_cols =cols [2 :end ]
    return cand_cols ,total_valid ,informal 


def party_numeric_cols (df :pd .DataFrame )->Tuple [List [str ],str ,str ]:
    cols =list (df .columns )
    total_valid =next ((c for c in cols if str (c ).strip ().lower ()=="total valid party votes"),cols [-2 ])
    informal =next ((c for c in cols if "informal"in str (c ).strip ().lower ()),cols [-1 ])
    idx_total_valid =cols .index (total_valid )
    idx_informal =cols .index (informal )
    end =min (idx_total_valid ,idx_informal )
    party_cols =cols [2 :end ]
    return party_cols ,total_valid ,informal 


def totals_triplet (valid :float ,informal :float )->Dict [str ,Number ]:
    valid =float (valid )
    informal =float (informal )
    return {"valid":valid ,"informal":informal ,"valid_plus_informal":valid +informal }


def fail_kv (key :str ,qa :float ,official :float )->Dict [str ,Any ]:
    qa =float (qa );official =float (official )
    diff =official -qa 
    pct =(diff /qa *100 )if qa !=0 else (0 if official ==0 else float ("inf"))
    return {
    "key":key ,
    "qa_value":qa ,
    "official_value":official ,
    "diff_official_minus_qa":diff ,
    "pct_diff_of_qa":pct ,
    }


    # ---------------- roster extraction ----------------

def extract_candidate_roster (df :pd .DataFrame ,totals_idx :Optional [int ])->pd .DataFrame :
# Candidate roster section (2002-2023) is typically:
#   candidate name | party | total candidate votes | ...
# and appears after the voting-place table + totals row.
    if totals_idx is None or df .shape [1 ]<3 :
        return pd .DataFrame (columns =["candidate","party","total_candidate_votes"])

    c0 ,c1 ,c2 =df .columns [0 ],df .columns [1 ],df .columns [2 ]

    start =totals_idx +1 
    marker =None 
    for i in range (start ,df .shape [0 ]):
        v0 =str (df .iloc [i ][c0 ]).strip ().lower ()
        if "electorate candidate valid votes"in v0 :
            marker =i 
            break 
    if marker is not None :
        start =marker +1 

    rows =[]
    for i in range (start ,df .shape [0 ]):
        cand =df .iloc [i ][c0 ]
        party =df .iloc [i ][c1 ]
        if pd .isna (cand )or str (cand ).strip ()in ["","nan"]:
            continue 
        tot_val =_to_num (pd .Series ([df .iloc [i ][c2 ]])).iloc [0 ]
        if pd .isna (tot_val ):
            found =None 
            for j in range (2 ,min (df .shape [1 ],6 )):
                v =_to_num (pd .Series ([df .iloc [i ][df .columns [j ]]])).iloc [0 ]
                if pd .notna (v ):
                    found =v 
                    break 
            if found is None :
                continue 
            tot_val =found 
        rows .append ({
        "candidate":str (cand ).strip (),
        "party":""if pd .isna (party )else str (party ).strip (),
        "total_candidate_votes":float (tot_val ),
        })
    return pd .DataFrame (rows )


def extract_party_roster (df :pd .DataFrame ,totals_idx :Optional [int ])->pd .DataFrame :
# Party roster section (2002-2023) is typically:
#   party name | total party votes | ...
# and appears after the voting-place table + totals row.
    if totals_idx is None or df .shape [1 ]<2 :
        return pd .DataFrame (columns =["party","total_party_votes"])

    c0 =df .columns [0 ]

    start =totals_idx +1 
    marker =None 
    for i in range (start ,df .shape [0 ]):
        v0 =str (df .iloc [i ][c0 ]).strip ().lower ()
        if "electorate party valid votes"in v0 :
            marker =i 
            break 
    if marker is not None :
        start =marker +1 

    rows =[]
    for i in range (start ,df .shape [0 ]):
        party =df .iloc [i ][c0 ]
        if pd .isna (party )or str (party ).strip ()in ["","nan"]:
            continue 
        found =None 
        for j in range (1 ,min (df .shape [1 ],6 )):
            v =_to_num (pd .Series ([df .iloc [i ][df .columns [j ]]])).iloc [0 ]
            if pd .notna (v ):
                found =v 
                break 
        if found is None :
            continue 
        rows .append ({"party":str (party ).strip (),"total_party_votes":float (found )})

    return pd .DataFrame (rows )


def port_candidate_roster_csv (candidate_csv :Path ,out_csv :Path )->None :
    """Emit a candidate roster CSV including party affiliations.

    Names + party affiliations are taken from the roster block after the totals row.
    Totals carried forward are QA totals computed from polling-place rows (NOT provided totals).
    """
    df =read_csv_atomic (candidate_csv )
    trow =find_totals_row (df )
    roster =extract_candidate_roster (df ,trow )

    # QA totals from rows, keyed by the main-table candidate column headers
    qa =_qa_candidate_totals_from_rows (candidate_csv )
    cand_cols ,_ ,_ =candidate_numeric_cols (df )

    def norm (s :str )->str :
        return re .sub (r"\s+"," ",str (s ).strip ()).casefold ()

    header_map ={norm (c ):float (qa .get (str (c ).strip (),0 ))for c in cand_cols }

    if not roster .empty :
        roster ["total_candidate_votes"]=roster ["candidate"].apply (lambda x :header_map .get (norm (x ),float ("nan")))
    roster =_format_df_numbers_for_csv (roster )
    out_csv .parent .mkdir (parents =True ,exist_ok =True )
    roster .to_csv (out_csv ,index =False ,encoding ="utf-8")

def port_party_roster_csv (party_csv :Path ,out_csv :Path )->None :
    """Emit a party roster CSV.

    Party names are taken from the main-table party vote columns.
    Totals carried forward are QA totals computed from polling-place rows (NOT provided totals).
    """
    df =read_csv_atomic (party_csv )
    party_cols ,_ ,_ =party_numeric_cols (df )
    qa =_qa_party_totals_from_rows (party_csv )

    rows =[{"party":str (c ).strip (),"total_party_votes":float (qa .get (str (c ).strip (),0 ))}for c in party_cols ]
    roster =pd .DataFrame (rows ,columns =["party","total_party_votes"])
    roster =_format_df_numbers_for_csv (roster )
    out_csv .parent .mkdir (parents =True ,exist_ok =True )
    roster .to_csv (out_csv ,index =False ,encoding ="utf-8")

def checksum_candidate_atomic_detailed (candidate_csv :Path )->Dict [str ,Any ]:
    df =read_csv_atomic (candidate_csv )
    trow =find_totals_row (df )

    cand_cols ,total_valid_col ,informal_col =candidate_numeric_cols (df )
    for c in cand_cols +[total_valid_col ,informal_col ]:
        df [c ]=_to_num (df [c ])

    result :Dict [str ,Any ]={
    "file":str (candidate_csv ),
    "totals":None ,
    "checks":{
    "voting_place_row_sum_vs_total_valid_column":{"passed":[],"failed":[]},
    "informal_column_sum_vs_totals":{"passed":[],"failed":[]},
    "totals_row_candidate_cols_vs_atomic_sums":{"passed":[],"failed":[]},
    "roster_candidate_totals_vs_atomic_sums":{"passed":[],"failed":[]},
    "valid_plus_informal_row_check":{"passed":[],"failed":[]},
    },
    }

    if trow is None :
        for k in result ["checks"].keys ():
            result ["checks"][k ]["failed"].append ({"key":"TOTALS_ROW_NOT_FOUND"})
        return result 

    atomic =df .iloc [:trow ].copy ()

    provided_total_valid =float (df .loc [trow ,total_valid_col ])if pd .notna (df .loc [trow ,total_valid_col ])else 0 
    provided_total_inf =float (df .loc [trow ,informal_col ])if pd .notna (df .loc [trow ,informal_col ])else 0 
    result ["totals"]=totals_triplet (provided_total_valid ,provided_total_inf )

    for i in range (atomic .shape [0 ]):
        loc =location_pair (df ,i )
        qa_valid =float (atomic .iloc [i ][cand_cols ].sum (skipna =True ))
        official_valid =float (atomic .iloc [i ][total_valid_col ])if pd .notna (atomic .iloc [i ][total_valid_col ])else 0 
        row_inf =float (atomic .iloc [i ][informal_col ])if pd .notna (atomic .iloc [i ][informal_col ])else 0 
        payload ={"location":loc ,"row_totals":totals_triplet (official_valid ,row_inf )}
        if qa_valid ==official_valid :
            result ["checks"]["voting_place_row_sum_vs_total_valid_column"]["passed"].append (payload )
        else :
            result ["checks"]["voting_place_row_sum_vs_total_valid_column"]["failed"].append ({**payload ,**fail_kv ("valid_row_sum",qa_valid ,official_valid )})

    atomic_sums =atomic [cand_cols +[total_valid_col ,informal_col ]].sum (numeric_only =True )

    for c in cand_cols +[total_valid_col ]:
        qa =float (atomic_sums [c ])if pd .notna (atomic_sums [c ])else 0 
        official =float (df .loc [trow ,c ])if pd .notna (df .loc [trow ,c ])else 0 
        if qa ==official :
            result ["checks"]["totals_row_candidate_cols_vs_atomic_sums"]["passed"].append ({"key":str (c )})
        else :
            result ["checks"]["totals_row_candidate_cols_vs_atomic_sums"]["failed"].append (fail_kv (str (c ),qa ,official ))

    qa_inf =float (atomic_sums [informal_col ])if pd .notna (atomic_sums [informal_col ])else 0 
    official_inf =float (df .loc [trow ,informal_col ])if pd .notna (df .loc [trow ,informal_col ])else 0 
    if qa_inf ==official_inf :
        result ["checks"]["informal_column_sum_vs_totals"]["passed"].append ({"key":informal_col })
    else :
        result ["checks"]["informal_column_sum_vs_totals"]["failed"].append (fail_kv (informal_col ,qa_inf ,official_inf ))

    qa_total_ballots =float (provided_total_valid +provided_total_inf )
    found =False 
    for j in range (trow +1 ,min (trow +6 ,df .shape [0 ])):
        tv =df .loc [j ,total_valid_col ]
        inf =df .loc [j ,informal_col ]
        if pd .notna (tv )and (pd .isna (inf )or float (inf )==0 ):
            found =True 
            official =float (tv )
            if official ==qa_total_ballots :
                result ["checks"]["valid_plus_informal_row_check"]["passed"].append ({"key":"valid_plus_informal"})
            else :
                result ["checks"]["valid_plus_informal_row_check"]["failed"].append (fail_kv ("valid_plus_informal",qa_total_ballots ,official ))
            break 
    if not found :
        result ["checks"]["valid_plus_informal_row_check"]["failed"].append ({"key":"ROW_AFTER_TOTALS_NOT_FOUND"})

    roster =extract_candidate_roster (df ,trow )
    atomic_by_candidate ={str (c ).strip ():float (atomic [c ].sum (skipna =True ))for c in cand_cols }
    if roster .empty :
        result ["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append ({"key":"ROSTER_NOT_FOUND"})
    else :
        for _ ,r in roster .iterrows ():
            name =str (r ["candidate"]).strip ()
            qa =atomic_by_candidate .get (name )
            official =float (r ["total_candidate_votes"])
            if qa is None :
                result ["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append ({"key":f"MISSING_HEADER:{name}"})
                continue 
            if qa ==official :
                result ["checks"]["roster_candidate_totals_vs_atomic_sums"]["passed"].append ({"key":name })
            else :
                result ["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append (fail_kv (name ,float (qa ),official ))

    return result 


def checksum_party_atomic_detailed (party_csv :Path )->Dict [str ,Any ]:
    df =read_csv_atomic (party_csv )
    trow =find_totals_row (df )

    party_cols ,total_valid_col ,informal_col =party_numeric_cols (df )
    for c in party_cols +[total_valid_col ,informal_col ]:
        df [c ]=_to_num (df [c ])

    result :Dict [str ,Any ]={
    "file":str (party_csv ),
    "totals":None ,
    "checks":{
    "voting_place_row_sum_vs_total_valid_column":{"passed":[],"failed":[]},
    "informal_column_sum_vs_totals":{"passed":[],"failed":[]},
    "totals_row_party_cols_vs_atomic_sums":{"passed":[],"failed":[]},
    "roster_party_totals_vs_atomic_sums":{"passed":[],"failed":[]},
    "valid_plus_informal_row_check":{"passed":[],"failed":[]},
    },
    }

    if trow is None :
        for k in result ["checks"].keys ():
            result ["checks"][k ]["failed"].append ({"key":"TOTALS_ROW_NOT_FOUND"})
        return result 

    atomic =df .iloc [:trow ].copy ()

    provided_total_valid =float (df .loc [trow ,total_valid_col ])if pd .notna (df .loc [trow ,total_valid_col ])else 0 
    provided_total_inf =float (df .loc [trow ,informal_col ])if pd .notna (df .loc [trow ,informal_col ])else 0 
    result ["totals"]=totals_triplet (provided_total_valid ,provided_total_inf )

    for i in range (atomic .shape [0 ]):
        loc =location_pair (df ,i )
        qa_valid =float (atomic .iloc [i ][party_cols ].sum (skipna =True ))
        official_valid =float (atomic .iloc [i ][total_valid_col ])if pd .notna (atomic .iloc [i ][total_valid_col ])else 0 
        row_inf =float (atomic .iloc [i ][informal_col ])if pd .notna (atomic .iloc [i ][informal_col ])else 0 
        payload ={"location":loc ,"row_totals":totals_triplet (official_valid ,row_inf )}
        if qa_valid ==official_valid :
            result ["checks"]["voting_place_row_sum_vs_total_valid_column"]["passed"].append (payload )
        else :
            result ["checks"]["voting_place_row_sum_vs_total_valid_column"]["failed"].append ({**payload ,**fail_kv ("valid_row_sum",qa_valid ,official_valid )})

    atomic_sums =atomic [party_cols +[total_valid_col ,informal_col ]].sum (numeric_only =True )

    for c in party_cols +[total_valid_col ]:
        qa =float (atomic_sums [c ])if pd .notna (atomic_sums [c ])else 0 
        official =float (df .loc [trow ,c ])if pd .notna (df .loc [trow ,c ])else 0 
        if qa ==official :
            result ["checks"]["totals_row_party_cols_vs_atomic_sums"]["passed"].append ({"key":str (c )})
        else :
            result ["checks"]["totals_row_party_cols_vs_atomic_sums"]["failed"].append (fail_kv (str (c ),qa ,official ))

    qa_inf =float (atomic_sums [informal_col ])if pd .notna (atomic_sums [informal_col ])else 0 
    official_inf =float (df .loc [trow ,informal_col ])if pd .notna (df .loc [trow ,informal_col ])else 0 
    if qa_inf ==official_inf :
        result ["checks"]["informal_column_sum_vs_totals"]["passed"].append ({"key":informal_col })
    else :
        result ["checks"]["informal_column_sum_vs_totals"]["failed"].append (fail_kv (informal_col ,qa_inf ,official_inf ))

    qa_total_ballots =float (provided_total_valid +provided_total_inf )
    found =False 
    for j in range (trow +1 ,min (trow +6 ,df .shape [0 ])):
        tv =df .loc [j ,total_valid_col ]
        inf =df .loc [j ,informal_col ]
        if pd .notna (tv )and (pd .isna (inf )or float (inf )==0 ):
            found =True 
            official =float (tv )
            if official ==qa_total_ballots :
                result ["checks"]["valid_plus_informal_row_check"]["passed"].append ({"key":"valid_plus_informal"})
            else :
                result ["checks"]["valid_plus_informal_row_check"]["failed"].append (fail_kv ("valid_plus_informal",qa_total_ballots ,official ))
            break 
    if not found :
        result ["checks"]["valid_plus_informal_row_check"]["failed"].append ({"key":"ROW_AFTER_TOTALS_NOT_FOUND"})

    roster =extract_party_roster (df ,trow )
    atomic_by_party ={str (c ).strip ():float (atomic [c ].sum (skipna =True ))for c in party_cols }
    if roster .empty :
        result ["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append ({"key":"ROSTER_NOT_FOUND"})
    else :
        for _ ,r in roster .iterrows ():
            name =str (r ["party"]).strip ()
            qa =atomic_by_party .get (name )
            official =float (r ["total_party_votes"])
            if qa is None :
                result ["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append ({"key":f"MISSING_HEADER:{name}"})
                continue 
            if qa ==official :
                result ["checks"]["roster_party_totals_vs_atomic_sums"]["passed"].append ({"key":name })
            else :
                result ["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append (fail_kv (name ,float (qa ),official ))

    return result 


    # ---------------- Split-vote endstate checksums (2002) ----------------

def _qa_candidate_totals_from_rows (candidate_csv :Path )->Dict [str ,float ]:
    """Compute QA totals by summing polling-place rows (NOT using provided totals row).

    Returns a dict keyed by the candidate column headers in the main table, plus:
      - informal totals under several common keys
      - total valid candidate votes (computed)
    """
    df =read_csv_atomic (candidate_csv )
    trow =find_totals_row (df )
    if trow is None :
    # best-effort: treat entire df as polling rows
        trow =df .shape [0 ]
    cand_cols ,total_valid_col ,informal_col =candidate_numeric_cols (df )
    # Coerce numeric columns on polling rows only
    poll =df .iloc [:trow ].copy ()
    for c in cand_cols +[total_valid_col ,informal_col ]:
        if c in poll .columns :
            poll [c ]=_to_num (poll [c ])
    out :Dict [str ,float ]={}
    for c in cand_cols :
        out [str (c ).strip ()]=float (poll [c ].sum (skipna =True ))if c in poll .columns else 0 
        # Informals total
    informal_total =float (poll [informal_col ].sum (skipna =True ))if informal_col in poll .columns else 0 
    out [str (informal_col ).strip ()]=informal_total 
    out ["Informals"]=informal_total 
    out ["Informal Candidate Votes"]=informal_total 
    # Total valid candidate votes (QA): sum of candidate columns
    total_valid =float (sum (out [str (c ).strip ()]for c in cand_cols ))if cand_cols else 0 
    out [str (total_valid_col ).strip ()]=total_valid 
    out ["Total valid candidate votes"]=total_valid 
    return out 


def _qa_party_totals_from_rows (party_csv :Path )->Dict [str ,float ]:
    """Compute QA party totals by summing polling-place rows (NOT using provided totals row)."""
    df =read_csv_atomic (party_csv )
    trow =find_totals_row (df )
    if trow is None :
        trow =df .shape [0 ]
    party_cols ,total_valid_col ,informal_col =party_numeric_cols (df )
    poll =df .iloc [:trow ].copy ()
    for c in party_cols +[total_valid_col ,informal_col ]:
        if c in poll .columns :
            poll [c ]=_to_num (poll [c ])
    out :Dict [str ,float ]={}
    for c in party_cols :
        out [str (c ).strip ()]=float (poll [c ].sum (skipna =True ))if c in poll .columns else 0 
        # Informals totals (party informals) can be useful in diagnostics
    informal_total =float (poll [informal_col ].sum (skipna =True ))if informal_col in poll .columns else 0 
    out [str (informal_col ).strip ()]=informal_total 
    out ["Informals"]=informal_total 
    # Total valid party votes (QA): sum of party columns
    total_valid =float (sum (out [str (c ).strip ()]for c in party_cols ))if party_cols else 0 
    out [str (total_valid_col ).strip ()]=total_valid 
    out ["Total valid party votes"]=total_valid 
    return out 


def _provided_candidate_totals_from_totals_row (candidate_csv :Path )->Dict [str ,float ]:
    """Read PROVIDED totals from the totals row (for integrity comparison only)."""
    df =read_csv_atomic (candidate_csv )
    trow =find_totals_row (df )
    if trow is None :
        return {}
    cand_cols ,total_valid_col ,informal_col =candidate_numeric_cols (df )
    for c in cand_cols +[total_valid_col ,informal_col ]:
        df [c ]=_to_num (df [c ])
    out ={str (c ).strip ():float (df .loc [trow ,c ])if pd .notna (df .loc [trow ,c ])else 0 for c in cand_cols }
    informal_total =float (df .loc [trow ,informal_col ])if pd .notna (df .loc [trow ,informal_col ])else 0 
    out [str (informal_col ).strip ()]=informal_total 
    out ["Informals"]=informal_total 
    out ["Informal Candidate Votes"]=informal_total 
    total_valid =float (df .loc [trow ,total_valid_col ])if pd .notna (df .loc [trow ,total_valid_col ])else 0 
    out [str (total_valid_col ).strip ()]=total_valid 
    out ["Total valid candidate votes"]=total_valid 
    return out 


def _provided_party_totals_from_totals_row (party_csv :Path )->Dict [str ,float ]:
    """Read PROVIDED totals from the totals row (for integrity comparison only)."""
    df =read_csv_atomic (party_csv )
    trow =find_totals_row (df )
    if trow is None :
        return {}
    party_cols ,total_valid_col ,informal_col =party_numeric_cols (df )
    for c in party_cols +[total_valid_col ,informal_col ]:
        df [c ]=_to_num (df [c ])
    out ={str (c ).strip ():float (df .loc [trow ,c ])if pd .notna (df .loc [trow ,c ])else 0 for c in party_cols }
    informal_total =float (df .loc [trow ,informal_col ])if pd .notna (df .loc [trow ,informal_col ])else 0 
    out [str (informal_col ).strip ()]=informal_total 
    out ["Informals"]=informal_total 
    total_valid =float (df .loc [trow ,total_valid_col ])if pd .notna (df .loc [trow ,total_valid_col ])else 0 
    out [str (total_valid_col ).strip ()]=total_valid 
    out ["Total valid party votes"]=total_valid 
    return out 


def _read_atomic_candidate_totals (candidate_csv :Path )->Dict [str ,float ]:
    """Return QA totals computed from polling-place rows (NOT the provided totals row)."""
    return _qa_candidate_totals_from_rows (candidate_csv )

def _read_atomic_party_totals (party_csv :Path )->Dict [str ,float ]:
    """Return QA totals computed from polling-place rows (NOT the provided totals row)."""
    return _qa_party_totals_from_rows (party_csv )

def checksum_splitvote_endstate_2002 (
split_endstate_csv :Path ,
candidate_csv :Path ,
party_csv :Path ,
)->Dict [str ,Any ]:
    """2002-era splitvote endstate checks (delegated to tally_assure.eras.era_2002)."""
    from .eras .era_2002 .checks_2002 import checksum_splitvote_endstate_2002 as _impl 
    return _impl (split_endstate_csv =split_endstate_csv ,candidate_csv =candidate_csv ,party_csv =party_csv )
