---
toc: true
layout: post
description: Examples on how to use R with MicroStrategy in Jupyter
categories: [python, microstrategy, API]
title: MicroStrategy API with R
---

# MicroStrategy REST API with R

<a id='top'></a>
### MicroStrategy Reference Material:
*  MicroStrategy RESTful API Interactive (your Local Demo): http://yourmstrEnv.com/MicroStrategyLibrary/api-docs/
* [MicroStrategy RESTful API Interactive (external demo)](https://demo.microstrategy.com/MicroStrategyLibrary/api-docs/index.html#/)
* [MicroStrategy REST API Online Documentation](https://lw.microstrategy.com/msdz/MSDL/GARelease_Current/docs/projects/RESTSDK/Content/topics/REST_API/REST_API.htm)

### Additional Resources/Inspirations:
* [MicroStrategy Sample API Python Example by Robert Prochowicz](https://community.microstrategy.com/s/article/REST-API-10-9-code-example-in-Python)
* [Machine Learning with Python On-Demand Video with Scott Rigney](https://www.microstrategy.com/us/resources/library/webcasts/machine-learning-with-python-train-models-on-trust)

### R Library References:
* [httr](http://httr.r-lib.org/)
* [jsonlite](https://www.rdocumentation.org/packages/jsonlite/versions/1.5)

### List of Code Examples :
1. [login()](#auth)
2. [sessionValidate()](#test)
3. [userInfo()](#user)
4. [listProjcets()](#projects)    
5. [getLibrary()](#library)
6. [searchObjects()](#search)
7. [cubeObjects()](#cubeobjects)
8. [Logout User()](#exit)

---- WIP to be added -----
9.  [Cube Instance](#cubeinstance)
10. [Get Cube Data](#cubedata)
11. [Create Cube](#createcube)
12. [Write/publish a Cube](#writecube)


### Import `httr` and `jsonlite` libraries


```R
library(httr)
library(jsonlite)
```

## Set Parameters 

Create the necessary varibales such as `username`, `password`, `projectid` and `baseURL`


```R
username <- 'Administrator'
password <- ''
baseURL <- "http://youMstrEnv/MicroStrategyLibrary/api/"
projectId <- 'B19DEDCC11D4E0EFC000EB9495D0F44F'
```

<a id='auth'></a>
## Authentication: Returns authToken & SessionId
[Top](#top)

Implementation Notes (source: MicroStrategy Documentation):  
Authenticate a user and create an HTTP session on the web server where the user’s MicroStrategy sessions are stored. This request returns an authorization token (X-MSTR-AuthToken) which will be submitted with subsequent requests. The body of the request contains the information needed to create the session. The loginMode parameter in the body specifies the authentication mode to use. You can authenticate with one of the following authentication modes: Standard (1), Anonymous (8), or LDAP (16). Authentication modes can be enabled through the System Administration REST APIs, if they are supported by the deployment. If you are not able to authenticate using any of the authentication modes, please contact your administrator to determine current support or currently enabled authentication modes.



```R
login <- function(baseURL, username, password) {
    d <- list(username = username,
             password = password,
             loginMode = '1')
    
    r <- POST(paste(base_url , 'auth/login', sep = ''), query = d)
    
    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)

    if (http_status(r)$category == 'Success') {
      cat("Success")
      authToken <- httpheader$'x-mstr-authtoken'
      sessionCoookies <- httpcookies$value
      cat("\nauthToken :", authToken)
      cat("\nSessionCookie:", sessionCoookies)
      authList <- list("authToken"=authToken, "sessionId"=sessionCoookies)  
      return(authList)  
    } else {
      cat(httpstats$category, httpstats$reason, httpstats$message)
    }
}
```


```R
#return a list
auth <- login(baseURL, username, password) 
```

    Success
    authToken : k4vchhmjv7me39v9816gv3den2
    SessionCookie: 18E1E53ADDBA549D077EACE563C208AB

<a id='test'></a>
## Test Session
[Top](#top)

Implementation Notes (source: MicroStrategy Documentation):  
Get information about a configuration session. You obtain the authorization token needed to execute the request using POST /auth/login; you pass the authorization token in the request header. Each time you call this endpoint, both the HTTP and Intelligence Server session timeouts are reset. This request returns information about the authenticated user, locale, timeout duration, maximum number of concurrent searches, and limit on number of instances kept in memory.


```R
sessionValidate <- function(baseURL, auth){
    
    r <- GET(paste(base_url , 'sessions', sep = ''), 
             add_headers('X-MSTR-AuthToken'=auth$authToken, Accept = 'application/json', cookies=auth$sessionId))
    
    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)

    if (http_status(r)$category == 'Success') {
      print(toJSON(content(r)))
    }  else {
      cat(httpstats$category, httpstats$reason, httpstats$message)   
    }
}
```


```R
sessionValidate(baseURL, auth)
```

    {"locale":[1033],"maxSearch":[3],"workingSet":[10],"timeout":[600],"id":["54F3D26011D2896560009A8E67019608"],"fullName":["Administrator"],"initials":["A"]} 


<a id='user'></a>
## Get UserInfo
[Top](#top)


```R
userInfo <- function(baseURL, auth){
    
     r <- GET(paste(baseURL , 'sessions/userInfo', sep = ''), 
             add_headers('X-MSTR-AuthToken'=auth$authToken, Accept = 'application/json', cookies=auth$sessionId))
    
    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)

    if (http_status(r)$category == 'Success') {
       a <- data.frame(content(r))
       return(a)
    }  else {
      cat(httpstats$category, httpstats$reason, httpstats$message)   
    }
    
}
```


```R
# Returns a data.frame object

user <- userInfo(baseURL, auth)
user
```


<table>
<thead><tr><th scope=col>metadataUser</th><th scope=col>id</th><th scope=col>fullName</th><th scope=col>initials</th></tr></thead>
<tbody>
	<tr><td>TRUE                            </td><td>54F3D26011D2896560009A8E67019608</td><td>Administrator                   </td><td>A                               </td></tr>
</tbody>
</table>



<a id='library'></a>
## Get Library for user
[Top](#top)

Implementation Notes (source: MicroStrategy Documentation)  
Get the library for the authenticated user. You obtain the authorization token needed to execute the request using POST /auth/login; you pass the authorization token in the request header.


```R
getLibrary <- function(baseURL, auth, flag) {
    
     r <- GET(paste(baseURL , 'library?outputFlag=', flag, sep = ''), 
             add_headers('X-MSTR-AuthToken'=auth$authToken, Accept = 'application/json', cookies=auth$sessionId))
    
    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)

    if (http_status(r)$category == 'Success') {
       a <- fromJSON(toJSON(content(r)))[c('id', 'name', 'projectId', 'active','lastViewedTime')]
       if (flag == 'DEFAULT'){
           b <- 0
           for (i in fromJSON(toJSON(content(r)))[c('target')]){
             b<- (i[,'id'])
             b <- data.frame(matrix(b, byrow = T), stringsAsFactors=FALSE)
             colnames(b) <- "targetId"
           }
         a$targetId <- b$targetId
       } 
       return(a)
    }  else {
      cat(httpstats$category, httpstats$reason, httpstats$message)   
    }
    
}
```


```R
# Return a data.frame object
libraryInfo <- getLibrary(baseURL, auth, 'DEFAULT')
libraryInfo
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>name</th><th scope=col>projectId</th><th scope=col>active</th><th scope=col>lastViewedTime</th><th scope=col>targetId</th></tr></thead>
<tbody>
	<tr><td>1B979449411E30E4E4502F918158EA40</td><td>Category Analysis               </td><td>B19DEDCC11D4E0EFC000EB9495D0F44F</td><td>TRUE                            </td><td>2018-08-11T07:49:40.000+0000    </td><td>512EDAA1487128DBBCA43E8525E10A11</td></tr>
	<tr><td>21A521BA4DB47ADAEBE19E9E9F7EC7D9    </td><td>Executive Business User Data Dossier</td><td>B19DEDCC11D4E0EFC000EB9495D0F44F    </td><td>TRUE                                </td><td>2018-08-10T19:36:00.000+0000        </td><td>FC6E8B6F4950540FC3595093E0FBA306    </td></tr>
	<tr><td>80AFEAD447DE2430F7E41FBB1B1EFCBA</td><td>Category Breakdown Dossier      </td><td>B19DEDCC11D4E0EFC000EB9495D0F44F</td><td>TRUE                            </td><td>2018-08-10T21:36:32.000+0000    </td><td>95005DFF4C4829CF5EE6E98877726566</td></tr>
</tbody>
</table>



<a id='projects'></a>
## List of Projects
[Top](#top)

Implementation Notes (Source: MicroStrategy Documentation)  
Get a list of projects which the authenticated user has access to. This returns the name, ID, description, alias, and status of each project; the status corresponds to values from EnumDSSXMLProjectStatus. You obtain the authorization token needed to execute the request using POST /auth/login; you pass the authorization token in the request header.


```R
listProjects <- function(baseURL, auth) {
    
     r <- GET(paste(baseURL , 'projects', sep = ''), 
             add_headers('X-MSTR-AuthToken'=auth$authToken, Accept = 'application/json', cookies=auth$sessionId))
    
    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)
    
    if (http_status(r)$category == 'Success') {
       a <- fromJSON(toJSON(content(r)))[c('id','name','description', 'status')]
       return(a)
    }  else {
      cat(httpstats$category, httpstats$reason, httpstats$message)   
    }

}
```


```R
# Return a data.frame object
projectList <- listProjects(baseURL, auth)
projectList
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>name</th><th scope=col>description</th><th scope=col>status</th></tr></thead>
<tbody>
	<tr><td>B19DEDCC11D4E0EFC000EB9495D0F44F                                                                                                                                                                                                                 </td><td>MicroStrategy Tutorial                                                                                                                                                                                                                           </td><td>MicroStrategy Tutorial project and application set designed to illustrate the platform's rich functionality. The theme is an Electronics, Books, Movies and Music store. Employees, Inventory, Finance, Product Sales and Suppliers are analyzed.</td><td>0                                                                                                                                                                                                                                                </td></tr>
	<tr><td>AF09B3E3458F78B4FBE4DEB68528BF7B                                                                                                                                                                                    </td><td>Human Resources Analysis Module                                                                                                                                                                                     </td><td>The Human Resources Analysis Module analyses workforce headcount, trends and profiles, employee attrition and recruitment, compensation and benefit costs and employee qualifications, performance and satisfaction.</td><td>0                                                                                                                                                                                                                   </td></tr>
	<tr><td>4DD3B04B40D227471401609D630C76ED</td><td>Enterprise Manager              </td><td>                                </td><td>0                               </td></tr>
</tbody>
</table>



<a id='search'></a>
## Search Objects
[Top](#top)

Implementation Notes (Source: MicroStrategy Documentation)  
Use the stored results of the Quick Search engine to return search results and display them as a list. The Quick Search engine periodically indexes the metadata and stores the results in memory, making Quick Search very fast but with results that may not be the most recent. You obtain the authorization token needed to execute the request using POST /auth/login. You identify the project by specifying the project ID in the request header; you obtain the project ID using GET /projects. You specify the search criteria using query parameters in the request; criteria can include the root folder ID, the search domain, the type of object, whether to return ancestors of the object, and a search pattern such as Begins With or Exactly. You use the offset and limit query parameters to control paging behavior. The offset parameter specifies where to start returning search results, and the limit parameter specifies how many results to return.


```R
searchObjects <- function(baseURL, auth, projectId, stype) {
    
    r <- GET(paste(baseURL , 'searches/results?type=', stype, sep = ''), 
             add_headers('X-MSTR-AuthToken'=auth$authToken, 'X-MSTR-ProjectID'=projectId,
                         Accept = 'application/json', cookies=auth$sessionId))

    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)
    
    if (http_status(r)$category == 'Success') {
       tmp <-content(r)$result
       df <- as.data.frame(tmp[[1]])
        for (i in 2:length(tmp)){
            df <- rbind(df, as.data.frame(tmp[[i]]))
        }
       return(df)
    }  else {
      cat(httpstats$category, httpstats$reason, httpstats$message)   
    }
    
}

```


```R
# Return a data.frame object
mySearch <- searchObjects(baseURL, auth, projectId, '39')
head(mySearch)
```


<table>
<thead><tr><th scope=col>name</th><th scope=col>id</th><th scope=col>type</th><th scope=col>subtype</th><th scope=col>extType</th><th scope=col>dateCreated</th><th scope=col>dateModified</th><th scope=col>version</th><th scope=col>acg</th><th scope=col>owner.name</th><th scope=col>owner.id</th></tr></thead>
<tbody>
	<tr><td>Search for all objects of type Grid       </td><td>87F09D2EBB9B462CAC4581ABCAD97BBD          </td><td>39                                        </td><td>9984                                      </td><td>0                                         </td><td>2005-06-27T21:33:41.000+0000              </td><td>2010-09-13T10:40:53.000+0000              </td><td>08B3974B493CE1E84106EB825B71CB6A          </td><td>255                                       </td><td>Administrator                             </td><td>54F3D26011D2896560009A8E67019608          </td></tr>
	<tr><td>Search for all objects of type Text Prompt</td><td>8A7CAF697BB64191BA3E15FA10DEDA61          </td><td>39                                        </td><td>9984                                      </td><td>0                                         </td><td>2005-06-27T21:33:42.000+0000              </td><td>2009-02-23T13:33:46.000+0000              </td><td>AC6316004E27925A85DDDF928D276A43          </td><td>255                                       </td><td>Administrator                             </td><td>54F3D26011D2896560009A8E67019608          </td></tr>
	<tr><td>MicroStrategy Web User Objects            </td><td>9F4A56074EDD734CBEFFC79A68BC36AF          </td><td>39                                        </td><td>9984                                      </td><td>0                                         </td><td>2010-04-12T11:13:59.000+0000              </td><td>2010-04-12T11:14:31.000+0000              </td><td>5726EAF84C05E5B3854423A0E8BA1106          </td><td>255                                       </td><td>Administrator                             </td><td>54F3D26011D2896560009A8E67019608          </td></tr>
	<tr><td>Search for all objects of type Hierarchy  </td><td>A1468ECD38754F90B56B611635AC550E          </td><td>39                                        </td><td>9984                                      </td><td>0                                         </td><td>2005-06-27T21:33:39.000+0000              </td><td>2009-02-23T13:33:43.000+0000              </td><td>5E82734349853883096289A9CE83F9A2          </td><td>255                                       </td><td>Administrator                             </td><td>54F3D26011D2896560009A8E67019608          </td></tr>
	<tr><td>Search for all objects of type Column     </td><td>57048C8A11D437E2C000039187BD3A4F          </td><td>39                                        </td><td>9984                                      </td><td>0                                         </td><td>2001-01-02T20:46:32.000+0000              </td><td>2007-03-04T16:42:01.000+0000              </td><td>9F27DD6B4FBED44E68CB869371E61BCA          </td><td>255                                       </td><td>Administrator                             </td><td>54F3D26011D2896560009A8E67019608          </td></tr>
	<tr><td>Search for all objects of type Document   </td><td>57048CAE11D437E2C000039187BD3A4F          </td><td>39                                        </td><td>9984                                      </td><td>0                                         </td><td>2001-01-02T20:46:30.000+0000              </td><td>2008-01-21T16:10:31.000+0000              </td><td>398B629141FCDB835E2CEA9D72D990B1          </td><td>255                                       </td><td>Administrator                             </td><td>54F3D26011D2896560009A8E67019608          </td></tr>
</tbody>
</table>



<a id='cubeobjects'></a>
## List Cube Objects
[Top](#top)

(mplementation Notes (Source: MicroStrategy Documentation)  
Get the definition of a specific cube, including attributes and metrics. The cube can be either an Intelligent Cube or a Direct Data Access (DDA)/MDX cube. The in-memory cube definition provides information about all available objects without actually running any data query/report. The results can be used by other requests to help filter large datasets and retrieve values dynamically, helping with performance and scalability. You obtain the authorization token needed to execute the request using POST /auth/login; you pass the authorization token and the project ID in the request header. You specify the cube ID in the path of the request; this can be either an Intelligent cube ID or a DDA/MDX cube ID.


```R
cubeObjects <- function(baseURL, auth, projectId, cubeId){
    
    r <- GET(paste(baseURL , 'cubes/', cubeId, sep = ''), 
             add_headers('X-MSTR-AuthToken'=auth$authToken, 'X-MSTR-ProjectID'=projectId,
                         Accept = 'application/json', cookies=auth$sessionId))
    
    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)
    
     if (http_status(r)$category == 'Success') {
       tmp <-content(r)$result
       mtrcs <- tmp$definition$availableObjects$metrics
       attr <- tmp$definition$availableObjects$attributes  
       mna <- rbind(do.call("rbind", attr)[, 1:3],do.call("rbind", mtrcs))
       return(mna)
    }  else {
      cat(httpstats$category, httpstats$reason, httpstats$message)   
    }
    
    
}

```


```R
# Return a data.frame object
cObjects <- cubeObjects(baseURL, auth, projectId, 'BD23848347017FC2C0B4509AED1AF7B4')
cObjects
```


<table>
<thead><tr><th scope=col>name</th><th scope=col>id</th><th scope=col>type</th></tr></thead>
<tbody>
	<tr><td>Country                         </td><td>8D679D3811D3E4981000E787EC6DE8A4</td><td>Attribute                       </td></tr>
	<tr><td>Catalog                         </td><td>8D679D3611D3E4981000E787EC6DE8A4</td><td>Attribute                       </td></tr>
	<tr><td>Category                        </td><td>8D679D3711D3E4981000E787EC6DE8A4</td><td>Attribute                       </td></tr>
	<tr><td>Subcategory                     </td><td>8D679D4F11D3E4981000E787EC6DE8A4</td><td>Attribute                       </td></tr>
	<tr><td>Cost                            </td><td>7FD5B69611D5AC76C000D98A4CC5F24F</td><td>Metric                          </td></tr>
	<tr><td>Gross Revenue                   </td><td>150349F04560BBA2592D019726DF77DD</td><td>Metric                          </td></tr>
	<tr><td>Units Sold                      </td><td>4C05190A11D3E877C000B3B2D86C964F</td><td>Metric                          </td></tr>
</tbody>
</table>



<a id='exit'></a>
##  Log Out and end session
[Top](#top)


```R
logout <- function(baseURL, auth){
     r <- GET(paste(base_url , 'auth/logout', sep = ''), 
             add_headers('X-MSTR-AuthToken'=auth$authToken, Accept = 'application/json', cookies=auth$sessionId))
    httpstats <- http_status(r)
    httpheader <- headers(r)
    httpcookies <- cookies(r)
    
    if (http_status(r)$category == 'Success') {
        print("Logged Out")
    } else {
        print(httpstats$category, httpstats$reason, httpstats$message) 
    }
    
}
```


```R
logout(baseURL, auth)
```

    [1] "Logged Out"

