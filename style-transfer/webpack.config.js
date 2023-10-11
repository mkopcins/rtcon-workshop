// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('path');

module.exports = () => {
    return {
        target: ['web'],
        entry: path.resolve(__dirname, 'main.js'),
        output: {
            path: path.resolve(__dirname, 'build'),
            filename: 'bundle.min.js',
            library: {
                type: 'umd'
            }
        },
        mode: 'production',
        watch: true
    }
};
