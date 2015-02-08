/*!
	\file version.h
	\brief Version definitions header.
	\author AG
*/

#ifndef CZ_VERSION_H
#define CZ_VERSION_H

#include "build.h"

/*!
	\name Application version information.
*/
/*@{*/
#define CZ_VER_MAJOR		0		/*!< Application major version number. */
#define CZ_VER_MINOR		5		/*!< Application minor version number. */
#define CZ_VER_STRING		"0.5"		/*!< Application version string. */

//#define CZ_VER_STATE		"SVN"
					 	/*!< Application release state. E.g.
						     - nondefined / commented out - stable release version.
						     - "Beta" - beta version/release candidate.
						     - "Alpha" - alpha version.
						     - "SVN" / "CVS" - versioning system tree snapshot. */

#ifdef CZ_VER_BUILD
#define CZ_VER_STRING_BUILD	CZ_VER_STRING "." CZ_VER_BUILD_STRING
#else
#define CZ_VER_STRING_BUILD	CZ_VER_STRING
#endif//CZ_VER_BUILD

/*!
	\def CZ_VERSION
	\brief Application version final string.
*/
#ifdef CZ_VER_STATE
#define CZ_VERSION		CZ_VER_STRING_BUILD " " CZ_VER_STATE
#else
#define CZ_VERSION		CZ_VER_STRING_BUILD
#endif//CZ_VER_STATE

#define CZ_NAME_SHORT		"CUDA-Z"	/*!< Application short name. */
#define CZ_NAME_LONG		"CUDA Information Utility"
						/*!< Application long name. */

#define CZ_DATE			__DATE__	/*!< Application compile date. */
#define CZ_TIME			__TIME__	/*!< Application compile time. */

#define CZ_ORG_NAME		"AG"		/*!< Organization name. */
#define CZ_ORG_DOMAIN		"cuda-z.sourceforge.net"
						/*!< Organization domain name. */
#define CZ_ORG_URL_MAINPAGE	"http://" CZ_ORG_DOMAIN "/"
						/*!< URL of main web site. */
#define CZ_ORG_URL_PROJECT	"http://sourceforge.net/projects/cuda-z/"
						/*!< URL of project page. */

#define CZ_COPY_INFO		"This software is distributed under the terms of the GNU General Public License Version 2."
						/*!< Program's copyright information. */
/*@}*/

#endif//CZ_VERSION_H
