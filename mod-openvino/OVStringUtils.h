// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once
#include <string>
#include <locale>
#ifdef _WIN32 
#include <windows.h>
#else
#include <codecvt>
#endif

static inline std::string FullPath(std::string base_dir, std::string filename)
{
#ifdef WIN32
   const std::string os_sep = "\\";
#else
   const std::string os_sep = "/";
#endif
   return base_dir + os_sep + filename;
}

static inline std::string wstring_to_string(const std::wstring& wstr) {
#ifdef _WIN32 
   int size_needed = WideCharToMultiByte(CP_ACP, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
   std::string strTo(size_needed, 0);
   WideCharToMultiByte(CP_ACP, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
   return strTo;
#else 
   std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
   return wstring_decoder.to_bytes(wstr);
#endif 
}

// WA for OpenVINO locale caching issue (https://github.com/openvinotoolkit/openvino/issues/24370)
// RAII-based method for performing the following
// 1. Upon instantiation, changes global numeric locale to "C".
// 2. When instance goes out of scope, previous locale is restored.
class OVLocaleWorkaround
{
public:

   OVLocaleWorkaround()
   {
      //get the current locale
      char *current_locale = setlocale(LC_NUMERIC, NULL);
      if (current_locale)
      {
         _prev_locale = current_locale;
      }

      setlocale(LC_NUMERIC, "C");
   }

   ~OVLocaleWorkaround()
   {
      //set previous locale
      if (!_prev_locale.empty())
      {
         setlocale(LC_NUMERIC, _prev_locale.c_str());
      }
   }

private:

   std::string _prev_locale;

};
