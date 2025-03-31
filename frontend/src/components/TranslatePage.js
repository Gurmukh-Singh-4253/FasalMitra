import React, { useEffect } from 'react';

function TranslatePage() {
  useEffect(() => {
    const addGoogleTranslateScript = () => {
      const script = document.createElement('script');
      script.type = 'text/javascript';
      script.src =
        '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
      document.body.appendChild(script);
    };

    const googleTranslateElementInit = () => {
      new window.google.translate.TranslateElement(
        { pageLanguage: 'en' },
        'google_translate_element'
      );
    };

    window.googleTranslateElementInit = googleTranslateElementInit; // Make it accessible globally

    addGoogleTranslateScript();

    // Cleanup function to remove the script when the component unmounts
    return () => {
      const script = document.querySelector(
        'script[src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"]'
      );
      if (script) {
        document.body.removeChild(script);
        delete window.googleTranslateElementInit;
      }
    };
  }, []);

  return (
    <div>
      <h1>My Web Page</h1>
      <p>Hello everybody!</p>
      <p>Translate this page:</p>
      <div id="google_translate_element"></div>
      <p>
        You can translate the content of this page by selecting a language in
        the select box.
      </p>
    </div>
  );
}

export default TranslatePage;